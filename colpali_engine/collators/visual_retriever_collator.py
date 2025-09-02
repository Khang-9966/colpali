from typing import Any, Dict, List, Union, cast

from PIL.Image import Image

from colpali_engine.models.idefics_2 import ColIdefics2Processor
from colpali_engine.models.paligemma import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
import torch

class VisualRetrieverCollator:
    """
    Collator for training vision retrieval models.
    """

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 700,
        add_suffix: bool = True,
    ):
        self.processor = processor
        self.image_token_id = None
        self.max_length = max_length
        self.suffix = ""

        if isinstance(self.processor, ColPaliProcessor) or isinstance(self.processor, ColIdefics2Processor):
            self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
                self.processor.tokenizer.additional_special_tokens.index("<image>")
            ]

        if isinstance(self.processor, ColPaliProcessor):
            if self.processor.tokenizer.padding_side != "right":
                print("Setting padding side to right")
                self.processor.tokenizer.padding_side = "right"

        if add_suffix:
            self.suffix = self.processor.tokenizer.pad_token * 10

    def __call__(
        self,
        examples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Collate function for the vision retriever associated to the collator's processor.
        """
        # Placeholders
        texts_query: Union[List[str], List[None], List[Union[str, None]]] = []  # some documents don't have a query
        texts_doc_query: Union[List[str], List[None], List[Union[str, None]]] = []  # some documents don't have a query
        texts_document: Union[List[str], List[None], List[Union[str, None]]] = []  # some documents don't have a query
        
        
        images: List[Image] = []
        neg_images: List[Image] = []

        if self.processor is None or not isinstance(self.processor, BaseVisualRetrieverProcessor):
            raise ValueError("Processor should be provided for vision collator.")

        # Process each example
        for example in examples:
            if example["image"] is None:
                texts_doc_query.append(example["query"])
                texts_document.append(example["text_document"])
            else:
                texts_query.append(example["query"])
                images.append(cast(Image, example["image"]))

            if "neg_image" in example and example["neg_image"] is not None:
                neg_images.append(cast(Image, example["neg_image"]))

        batch_doc = None
        if len(images) > 0:
            batch_doc = self.processor.process_images(
                images=images,
            )

        # Process the negative documents (if available)
        batch_neg_doc = None
        if len(neg_images) > 0:
            batch_neg_doc = self.processor.process_images(
                images=neg_images,
            )

        # Process the queries
        batch_query = None
        if len(texts_query+texts_doc_query) > 0:
            texts_query = cast(List[str], texts_query+texts_doc_query)
            batch_query = self.processor.process_queries(
                queries=texts_query,
                max_length=self.max_length,
                suffix=self.suffix,
            )
            

        # Process text documents
        batch_text_doc = None
        if len(texts_document) > 0:
            texts_document = cast(List[str], texts_document)
            batch_text_doc = self.processor.process_docs(
                docs=texts_document,
                max_length=self.max_length,
                suffix=self.suffix,
            )

        if batch_text_doc is not None and batch_doc is not None:
            if batch_doc['input_ids'].shape[1] > batch_text_doc['input_ids'].shape[1]:
                batch_text_doc = self.processor.process_docs(
                    docs=texts_document,
                    max_length=batch_doc['input_ids'].shape[1],
                    suffix=self.suffix,
                    padding="max_length"
                )
            else  :
                batch_doc = self.processor.process_images(
                    images=images,
                    max_length=batch_text_doc['input_ids'].shape[1],
                    padding="max_length"
                )

        
        # Combine all batches
        batch_all = {}

        if batch_doc is not None:
            batch_all.update({f"im_doc_{k}": v for k, v in batch_doc.items()})
            
        if batch_text_doc is not None:
            batch_all.update({f"text_doc_{k}": v for k, v in batch_text_doc.items()})

        
        # Add queries
        if batch_query is not None:
            batch_all.update({f"query_{k}": v for k, v in batch_query.items()})
            
        # Add negative documents
        if batch_neg_doc is not None:
            batch_all.update({f"neg_doc_{k}": v for k, v in batch_neg_doc.items()})
            
        return batch_all
