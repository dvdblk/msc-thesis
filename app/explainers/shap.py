import numpy as np
import pandas as pd
import shap
import torch

from shap.maskers import Text as TextMasker
from sklearn.feature_extraction.text import TfidfVectorizer

from app.explainers.base import BaseExplainer
from app.explainers.model import XAIOutput, ExplainerMethod


class TFIDFTextMasker(shap.maskers.Text):
    """Masker thaat masks out only the tokens that have higher than threshold TF-IDF scores."""

    def __init__(
        self, vectorizer_data, threshold=0.06, tokenizer=None, mask_value=None, **kwargs
    ):
        super().__init__(mask_token=mask_value, **kwargs)
        self.threshold = threshold

        # fit TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer.tokenize, token_pattern=".", max_df=0.96
        )
        vectorizer.fit_transform(vectorizer_data)
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer

    def __call__(self, mask, s, return_mask=False):
        """
        Args:
            mask: default mask given by the partition clustering (of length `n` tokens of tokenized `s`)
            s: input string
            return_mask: whether to return the mask or not (needed for CachedPartitionExplainer)
        """
        # TODO: is this necessary? probably not as the base value should not take into account the
        # masked tokens even in the base case. Which means this shouldn't be used...
        # is_computing_base_val = all(not value for value in mask)
        # if is_computing_base_val:
        #     return super().__call__(mask, s), mask

        # get a mask from the sample string s by thresholding on tf-idf values
        tokens = self.tokenizer.tokenize(s)

        token_scores = []
        dense_tokens = self.vectorizer.transform([s]).todense()
        vocabulary_dict = self.vectorizer.vocabulary_

        for token in tokens:
            score = dense_tokens[0, vocabulary_dict.get(token, -1)]
            token_scores.append(score)

        tf_idf_mask = np.array(token_scores) >= self.threshold

        # do a negation (bitwise not) because the final mask is negated
        # i.e. 'True' = never use "[MASK]" in this position which means
        # 'False' indicates that this token is of no importance thus won't be masked out
        tf_idf_mask = ~tf_idf_mask

        # make sure to pad new_mask to the same length as mask
        new_mask = np.pad(tf_idf_mask, (1, 1), mode="constant", constant_values=False)

        # do a bitwise or between new_mask and original mask
        new_mask = np.bitwise_or(new_mask, mask)

        # call the super class with the TF-IDF modified mask
        new_res = super().__call__(new_mask, s)

        if return_mask:
            return new_res, new_mask
        return new_res


class ShapExplainer(BaseExplainer):

    def __init__(self, model, tokenizer, device, xai_method, algorithm, masker=None):
        super().__init__(model, tokenizer, device, xai_method=xai_method)
        # Set the explainer for SHAP subclasses
        self.explainer = shap.Explainer(
            self._predictor_func,
            algorithm=algorithm,
            masker=masker or TextMasker(self.tokenizer),
            silent=True,
        )

    def _predictor_func(self, text):
        """Predictor function that SHAP uses while explaining"""
        # TODO: this could/should probably be model dependent? but works with SciBERT for now
        inputs = self.tokenizer(
            text.tolist(),
            padding=True,
            truncation=True,
            max_length=512,  # FIXME: this should be model dependent, SciBERT is 512
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits.cpu()

        # softmax
        probs = torch.exp(logits) / torch.exp(logits).sum(-1, keepdims=True)
        return probs

    def explain(self, abstract) -> XAIOutput:
        # Get SHAP values
        shap_values = self.explainer([abstract])

        # Get prediction probabilities and predicted_id with some tensor math
        # to avoid calling the model again...
        summed_shap_values = np.sum(shap_values.values, axis=1)
        probabilities = summed_shap_values + shap_values.base_values[0]
        predicted_id = np.argmax(probabilities)
        predicted_label = self.model.config.id2label[predicted_id]

        return XAIOutput(
            text=abstract,
            input_tokens=shap_values.data[
                0  # 0 is the index of the first (and only) sample (abstract) that was used
            ].tolist(),
            token_scores=shap_values.values[0].tolist(),
            predicted_id=predicted_id,
            predicted_label=predicted_label,
            probabilities=probabilities[0].tolist(),
            xai_method=self.xai_method,
            additional_values={
                "base_values": shap_values.base_values[0].tolist(),
            },
        )


class PartitionShapExplainer(ShapExplainer):
    def __init__(self, model, tokenizer, device):
        super().__init__(
            model,
            tokenizer,
            device,
            xai_method=ExplainerMethod.SHAP_PARTITION,
            algorithm="partition",
        )


class KernelShapExplainer(ShapExplainer):
    def __init__(self, model, tokenizer, device):
        super().__init__(
            model,
            tokenizer,
            device,
            xai_method=ExplainerMethod.SHAP_KERNEL,
            algorithm="kernel",
        )


class TfIdfPartitionShapExplainer(ShapExplainer):

    def __init__(self, tf_idf_data_path, model, tokenizer, device):
        # Load OSDG data
        osdg_data = pd.read_csv(tf_idf_data_path)

        super().__init__(
            model,
            tokenizer,
            device,
            xai_method=ExplainerMethod.SHAP_PARTITION_TFIDF,
            algorithm="partition",
            masker=TFIDFTextMasker(
                osdg_data["abstract"],
                tokenizer=tokenizer,
                threshold=0.06,
            ),
        )
