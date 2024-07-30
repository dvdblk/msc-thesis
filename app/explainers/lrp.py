import torch

from lxt.models.bert import attnlrp as bert_attnlrp, cp_lrp as bert_cplrp
from lxt.models.llama import attnlrp as llama_attnlrp, cp_lrp as llama_cplrp

from app.explainers.base import BaseExplainer
from app.explainers.model import XAIOutput, ExplainerMethod
from app.utils.tokenization import fix_bert_tokenization


class LRPExplainer(BaseExplainer):

    def __init__(self, model_family, model, tokenizer, device, max_seq_len, xai_method):
        """
        Args:
            model_family (str): Model family (e.g. "bert", "llama2", ...)
            model (torch.nn.Module): Model
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer
            device (torch.device): Device
        """
        super().__init__(model, tokenizer, device, max_seq_len, xai_method=xai_method)
        self.model_family = model_family

    def explain(self, abstract):
        input_ids = self.tokenizer(
            abstract,
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        if self.model_family == "scibert":
            inputs_embeds = self.model.bert.embeddings.word_embeddings(input_ids)
        elif (
            self.model_family == "llama2"
            or self.model_family == "llama3"
            or self.model_family == "unllama3"
        ):
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        n_classes = self.model.num_labels
        relevance_all = []

        predicted_id = None
        predicted_label = None
        probabilities = None

        for class_idx in range(n_classes):
            _inputs_embeds = inputs_embeds.clone().detach().requires_grad_()
            # forward pass
            logits = self.model(inputs_embeds=_inputs_embeds).logits

            if predicted_id is None:
                _, max_indices = torch.max(logits, dim=-1)

                predicted_id = max_indices.item()
                predicted_label = self.model.config.id2label[predicted_id]
                probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

            class_logits = logits[0, class_idx]
            class_logits.backward()
            relevance = _inputs_embeds.grad.float().sum(-1).cpu()[0]
            # normalize relevance between [-1, 1] for plotting
            relevance = relevance / relevance.abs().max()
            relevance_all.append(relevance)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens = fix_bert_tokenization(tokens)

        # turn relevance list into a tensor
        relevance_scores = torch.stack(relevance_all, dim=1)

        return XAIOutput(
            text=abstract,
            input_tokens=tokens,
            token_scores=relevance_scores.tolist(),
            predicted_id=predicted_id,
            predicted_label=predicted_label,
            probabilities=probabilities,
            xai_method=self.xai_method,
        )


class AttnLRPExplainer(LRPExplainer):
    """Attention LRP Explainer

    Note:
        Modifies model internals with LRP rules (can't use for training)
    """

    def __init__(self, model_family, model, tokenizer, device, max_seq_len):
        # TODO: remove LoRA layers if needed by merge_and_unload
        if model_family == "scibert":
            bert_attnlrp.register(model)
        elif (
            model_family == "llama2"
            or model_family == "llama3"
            or model_family == "unllama3"
        ):
            llama_attnlrp.register(model)
        else:
            raise ValueError(f"Unsupported model family for AttnLRP: {model_family}")
        super().__init__(
            model_family,
            model,
            tokenizer,
            device,
            max_seq_len=max_seq_len,
            xai_method=ExplainerMethod.ATTNLRP,
        )


class CPLRPExplainer(LRPExplainer):
    """Conservative Propagation Explainer

    Note:
        Modifies model internals with LRP rules (can't use for training)
    """

    def __init__(self, model_family, model, tokenizer, device, max_seq_len):
        # TODO: remove LoRA layers if needed by merge_and_unload
        if model_family == "scibert":
            bert_cplrp.register(model)
        elif (
            model_family == "llama2"
            or model_family == "llama3"
            or model_family == "unllama3"
        ):
            llama_cplrp.register(model)
        else:
            raise ValueError(f"Unsupported model family for CPLRP: {model_family}")
        super().__init__(
            model_family,
            model,
            tokenizer,
            device,
            max_seq_len=max_seq_len,
            xai_method=ExplainerMethod.CPLRP,
        )
