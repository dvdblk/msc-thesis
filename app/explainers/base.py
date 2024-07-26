import torch

from abc import ABC, abstractmethod


class BaseExplainer(ABC):
    """Base class for any XAI method"""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def explain(self, abstract):
        """
        Returns:
            dict: Explanation values (token scores, base values, input tokens)
        """
        raise NotImplementedError

    def __call__(self, abstract):
        return self.explain(abstract)

    def predictor(self, texts):
        # TODO: this could/should probably be model dependent? but works with SciBERT for now
        inputs = self.tokenizer(
            texts.tolist(),
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
