import torch
from captum.attr import InputXGradient, IntegratedGradients, Saliency

from app.explainers.base import BaseExplainer
from app.explainers.model import XAIOutput, ExplainerMethod
from app.utils.tokenization import fix_bert_tokenization


class GradientExplainer(BaseExplainer):
    def __init__(self, model, tokenizer, device, multiply_by_inputs=True):
        # xai_method = ExplainerMethod.GRADIENT_X_INPUT if multiply_by_inputs else ExplainerMethod.GRADIENT
        xai_method = ExplainerMethod.INPUTXGRADIENT
        super().__init__(model, tokenizer, device, xai_method=xai_method)
        self.multiply_by_inputs = multiply_by_inputs

    def _predictor_func(self, input_embeds):
        outputs = self.model(inputs_embeds=input_embeds)
        return outputs.logits

    def explain(self, abstract):
        inputs = self.tokenizer(
            abstract, max_length=512, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        input_embeds = self.model.get_input_embeddings()(inputs.input_ids)

        dl = (
            InputXGradient(self._predictor_func)
            if self.multiply_by_inputs
            else Saliency(self._predictor_func)
        )

        n_classes = self.model.num_labels
        input_len = inputs.attention_mask.sum().item()
        attr_all = torch.zeros((input_len, n_classes), device=self.device)

        for class_idx in range(n_classes):
            attr = dl.attribute(input_embeds, target=class_idx)
            attr = attr[0, :input_len, :].sum(-1)  # Pool over hidden size
            attr_all[:, class_idx] = attr

        logits = self._predictor_func(input_embeds)
        predicted_id = logits.argmax(dim=-1).item()
        predicted_label = self.model.config.id2label[predicted_id]
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()[0]

        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        tokens = fix_bert_tokenization(tokens)

        return XAIOutput(
            text=abstract,
            input_tokens=tokens,
            token_scores=attr_all.cpu().tolist(),
            predicted_id=predicted_id,
            predicted_label=predicted_label,
            probabilities=probabilities,
            xai_method=self.xai_method,
        )


class IntegratedGradientExplainer(BaseExplainer):
    def __init__(self, model, tokenizer, device, multiply_by_inputs=True):
        super().__init__(
            model, tokenizer, device, xai_method=ExplainerMethod.INTEGRATED_GRADIENT
        )
        self.multiply_by_inputs = multiply_by_inputs

    def _predictor_func(self, input_embeds):
        outputs = self.model(inputs_embeds=input_embeds)
        return outputs.logits

    def _generate_baselines(self, input_len):
        ids = (
            [self.tokenizer.cls_token_id]
            + [self.tokenizer.pad_token_id] * (input_len - 2)
            + [self.tokenizer.sep_token_id]
        )
        embeddings = self.model.get_input_embeddings()(
            torch.tensor(ids, device=self.device)
        )
        return embeddings.unsqueeze(0)

    def explain(self, abstract):
        inputs = self.tokenizer(
            abstract, max_length=512, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        input_embeds = self.model.get_input_embeddings()(inputs.input_ids)

        dl = IntegratedGradients(
            self._predictor_func, multiply_by_inputs=self.multiply_by_inputs
        )

        baselines = self._generate_baselines(inputs.attention_mask.sum().item())

        n_classes = self.model.num_labels
        input_len = inputs.attention_mask.sum().item()
        attr_all = torch.zeros((input_len, n_classes), device=self.device)

        for class_idx in range(n_classes):
            attr = dl.attribute(input_embeds, baselines=baselines, target=class_idx)
            attr = attr[0, :input_len, :].sum(-1)  # Pool over hidden size
            attr_all[:, class_idx] = attr

        logits = self._predictor_func(input_embeds)
        predicted_id = logits.argmax(dim=-1).item()
        predicted_label = self.model.config.id2label[predicted_id]
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()[0]

        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        tokens = fix_bert_tokenization(tokens)

        return XAIOutput(
            text=abstract,
            input_tokens=tokens,
            token_scores=attr_all.cpu().tolist(),
            predicted_id=predicted_id,
            predicted_label=predicted_label,
            probabilities=probabilities,
            xai_method=self.xai_method,
        )
