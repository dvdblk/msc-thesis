"""Contains Data classes and enumerations for XAI methods and their outputs (data models)"""

from typing import List, Optional
from enum import StrEnum, auto
from dataclasses import dataclass, field
from datetime import datetime, timezone


class ExplainerMethod(StrEnum):
    """Enumeration of available XAI methods"""

    def _generate_next_value_(name, start, count, last_values):
        return name.lower().replace("_", "-")

    SHAP_PARTITION = auto()
    SHAP_KERNEL = auto()
    SHAP_PARTITION_TFIDF = auto()
    ATTNLRP = auto()
    CPLRP = auto()
    LIME = auto()
    INTEGRATED_GRADIENT = auto()
    GRADIENTXINPUT = auto()


@dataclass
class XAIOutput:
    """Represents the output of an XAI method"""

    text: str
    """The input text"""
    input_tokens: List[str]
    """The input tokens"""
    token_scores: List[List[float]]
    """The relevancy / feature attribution scores of each token per class. 2D array of shape (num_tokens, num_classes)"""
    predicted_id: int
    """The predicted label index (with highest probability)"""
    predicted_label: str
    """The actual SDG label that was predicted (`predicted_id + 1` and stringified)"""
    probabilities: List[float]
    """The predicted probabilities for each label"""
    xai_method: ExplainerMethod
    """The XAI method used to generate the output"""
    additional_values: Optional[dict] = None
    """Additional values returned by the explainer (untyped, explainer specific)"""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """UTC timestamp of when this instance was created"""
