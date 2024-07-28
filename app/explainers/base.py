from abc import ABC, abstractmethod
from app.explainers.model import XAIOutput


class BaseExplainer(ABC):
    """Base class for any XAI method that provides explanations for a model prediction / decision"""

    def __init__(self, model, tokenizer, device, xai_method):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.xai_method = xai_method

    @abstractmethod
    def explain(self, abstract) -> XAIOutput:
        """
        Returns:
            XAIOutput: Explanation values (token scores, base values, input tokens)
        """
        raise NotImplementedError

    def __call__(self, abstract):
        return self.explain(abstract)
