import math
from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchFeature

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from .internvl.conversation import get_conv_template
from transformers import BatchFeature, ProcessorMixin

class ColInternVL2Processor(BaseVisualRetrieverProcessor, ProcessorMixin):
    """
    Processor for ColInternVL2.
    """
    attributes = [ "tokenizer"]
    image_processor_class = "InternVL2ImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    
    def __init__(self, tokenizer,  **kwargs):
        self.template = "Hermes-2"
        self.num_image_token = 256
        self.max_num = 6

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True, use_fast=False)
        else:
            self.tokenizer = tokenizer
            
        self.tokenizer.padding_side = 'left'
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
        self.IMG_START_TOKEN='<img>'
        self.IMG_END_TOKEN='</img>'
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        self.system_message = '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'
        super().__init__(tokenizer)
        
    # def from_pretrained(pretrained_model_name_or_path, template="Hermes-2", **kwargs):
    #     return ColInternVL2Processor(pretrained_model_name_or_path, template=template, **kwargs)
        
    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
    
        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    
        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    
        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def load_image(self, image, input_size=448, max_num=12):
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    
    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchFeature:
        """
        Process images for InternVl2.
        """

        pixel_values = [ self.load_image(image, max_num=self.max_num) for image in images]

        num_patches_list = [ pixel_.size(0) for pixel_ in pixel_values]
        image_flags = [ torch.tensor([1] * pixel_.shape[0], dtype=torch.long) for pixel_ in pixel_values ]
        
        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = "<image>\nDescribe the image."

            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + self.IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        model_inputs = self.tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'] #.to(self.device)
        attention_mask = model_inputs['attention_mask'] #.to(self.device)
        pixel_values = torch.cat(pixel_values)
        
        batch_doc = BatchFeature({
            "pixel_values" : pixel_values,
            "input_ids" : input_ids,
            "attention_mask" : attention_mask,
            # "image_flags" : image_flags
        })
        return batch_doc    
    
    def process_queries(
        self,
        queries: List[str],
        max_length: int = 100,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        """
        Process queries for InternVl2.
        """

        texts_query: List[str] = []

        for query in queries:
            query = f"Query: {query}"
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], query)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            texts_query.append(query)
            
        model_inputs = self.tokenizer(texts_query, return_tensors='pt', max_length=max_length, padding="longest")
        input_ids = model_inputs['input_ids'] #.to(self.device)
        attention_mask = model_inputs['attention_mask'] #.to(self.device)
        
        batch_query = BatchFeature({
            "pixel_values" : None,
            "input_ids" : input_ids,
            "attention_mask" : attention_mask,
        })
        return batch_query

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
    ) -> Tuple[int, int]:
        raise NotImplementedError("This method is not implemented for ColInternVL2.")
