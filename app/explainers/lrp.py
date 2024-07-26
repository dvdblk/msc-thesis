import torch

from app.explainers.base import BaseExplainer

from lxt.models.bert import attnlrp as bert_attnlrp
from lxt.models.llama import attnlrp as llama_attnlrp


class LRPExplainer(BaseExplainer):

    def __init__(self, model_family, model, tokenizer, device):
        """
        Args:
            model_family (str): Model family (e.g. "bert", "llama", ...)
            model (torch.nn.Module): Model
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer
            device (torch.device): Device
        """
        super().__init__(model, tokenizer, device)
        self.model_family = model_family

    def explain(self, abstract):
        input_ids = self.tokenizer(
            abstract, max_length=512, padding=True, truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        inputs_embeds = self.model.bert.get_input_embeddings()(input_ids)

        logits = self.model(inputs_embeds=inputs_embeds.requires_grad_()).logits

        # We explain the sequence label: acceptable or unacceptable
        max_logits, max_indices = torch.max(logits, dim=-1)

        # TODO: return predicted values and probs from here
        # out = self.model.config.id2label[max_indices.item()]

        max_logits.backward(max_logits)

        relevance = inputs_embeds.grad.float().sum(-1).cpu()[0]
        # normalize relevance between [-1, 1] for plotting
        relevance = relevance / relevance.abs().max()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        return {
            "token_scores": relevance.tolist(),
            "input_tokens": tokens,
        }


class AttnLRPExplainer(LRPExplainer):
    """Attention LRP Explainer

    Note:
        Modifies model internals with LRP rules (can't use for training)
    """

    def __init__(self, model_family, model, tokenizer, device):
        if model_family == "scibert":
            bert_attnlrp.register(model)
        elif model_family == "llama":
            llama_attnlrp.register(model)
        else:
            raise ValueError(f"Unsupported model family for AttnLRP: {model_family}")
        super().__init__(model_family, model, tokenizer, device)


class CPLRPExplainer(LRPExplainer):
    """Conservative Propagation Explainer

    Note:
        Modifies model internals with LRP rules (can't use for training)
    """

    def __init__(self, model_family, model, tokenizer, device):
        super().__init__(model_family, model, tokenizer, device)

    def explain(self, abstract):
        pass
