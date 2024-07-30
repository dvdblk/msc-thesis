import numpy as np
import torch
from lime.lime_text import LimeTextExplainer

from app.explainers.base import BaseExplainer
from app.explainers.model import XAIOutput, ExplainerMethod
from app.utils.tokenization import fix_bert_tokenization


class LimeExplainer(BaseExplainer):

    def __init__(
        self, model, tokenizer, device, max_seq_len, num_samples=650, max_samples=3000
    ):
        super().__init__(
            model, tokenizer, device, max_seq_len, xai_method=ExplainerMethod.LIME
        )
        self.num_samples = num_samples
        self.max_samples = max_samples
        self.lime_explainer = LimeTextExplainer(
            bow=False, class_names=list(model.config.id2label.values())
        )

    def _predictor_func(self, texts):
        all_probs = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            if outputs.logits.dtype == torch.bfloat16:
                outputs.logits = outputs.logits.float()

            logits = outputs.logits.cpu()
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs[0])

        return np.array(all_probs)

    def explain(self, abstract):
        tokens = self.tokenizer.tokenize(abstract)
        n_classes = self.model.num_labels
        n_tokens = len(tokens)

        n_samples = (
            min(len(tokens) ** 2, self.max_samples)
            if self.num_samples is None
            else self.num_samples
        )

        lime_explanation = self.lime_explainer.explain_instance(
            abstract,
            self._predictor_func,
            num_features=len(tokens),
            num_samples=n_samples,
            top_labels=n_classes,
        )

        predicted_id = lime_explanation.predict_proba.argmax()
        predicted_label = self.model.config.id2label[predicted_id]
        # TODO: remove the unnecessary list brackets here later to keep all methods in the same fmt
        # (this is done because the rest of the methods are in this 2D array format...)
        probabilities = [lime_explanation.predict_proba.tolist()]

        # Initialize token_scores with zeros
        token_scores = np.zeros((n_tokens, n_classes))

        for class_idx in range(n_classes):
            if class_idx in lime_explanation.local_exp:
                for token_idx, score in lime_explanation.local_exp[class_idx]:
                    token_scores[token_idx, class_idx] = score

        tokens = fix_bert_tokenization(tokens)

        return XAIOutput(
            text=abstract,
            input_tokens=tokens,
            token_scores=token_scores,
            predicted_id=predicted_id,
            predicted_label=predicted_label,
            probabilities=probabilities,
            xai_method=self.xai_method,
        )
