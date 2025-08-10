import torch
from transformers import Trainer


class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, is_vision_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.is_vision_model = is_vision_model  # Unused argument, will be removed in 0.4.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])

        has_im_doc = any(k.startswith("im_doc_") for k in inputs.keys())
        has_text_doc = "text_doc_input_ids" in inputs
        
        if has_im_doc and has_text_doc:
            im_doc_outputs = model(**{k[7:]: v for k, v in inputs.items() if k.startswith("im_doc_")})
            text_doc_outputs = model(input_ids=inputs["text_doc_input_ids"], attention_mask=inputs["text_doc_attention_mask"])
            doc_outputs = torch.cat((im_doc_outputs, text_doc_outputs), 0)
        elif has_im_doc:
            doc_outputs = model(**{k[7:]: v for k, v in inputs.items() if k.startswith("im_doc_")})
        elif has_text_doc:
            doc_outputs = model(input_ids=inputs["text_doc_input_ids"], attention_mask=inputs["text_doc_attention_mask"])

        
        if "neg_doc_input_ids" in inputs:
            neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
            loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
            return (loss, (query_outputs, doc_outputs, neg_doc_outputs)) if return_outputs else loss

        loss = self.loss_func(query_outputs, doc_outputs)
        return (loss, (query_outputs, doc_outputs)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")
        with torch.no_grad():
            query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
            
            has_im_doc = any(k.startswith("im_doc_") for k in inputs.keys())
            has_text_doc = "text_doc_input_ids" in inputs
            
            if has_im_doc and has_text_doc:
                im_doc_outputs = model(**{k[7:]: v for k, v in inputs.items() if k.startswith("im_doc_")})
                text_doc_outputs = model(input_ids=inputs["text_doc_input_ids"], attention_mask=inputs["text_doc_attention_mask"])
                doc_outputs = torch.cat((im_doc_outputs, text_doc_outputs), 0)
            elif has_im_doc:
                doc_outputs = model(**{k[7:]: v for k, v in inputs.items() if k.startswith("im_doc_")})
            elif has_text_doc:
                doc_outputs = model(input_ids=inputs["text_doc_input_ids"], attention_mask=inputs["text_doc_attention_mask"])
            
            if "neg_doc_input_ids" in inputs:
                neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
                loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
                return loss, None, None
            loss = self.loss_func(query_outputs, doc_outputs)
            return loss, None, None
