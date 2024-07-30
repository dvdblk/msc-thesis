from typing import List
from ferret.explainers.explanation import Explanation as FerretExplanation
from ferret.evaluators.faithfulness_measures import (
    AOPC_Comprehensiveness_Evaluation,
    AOPC_Sufficiency_Evaluation,
    TauLOO_Evaluation,
)
from ferret.evaluators.evaluation import (
    ExplanationEvaluation as FerretExplanationEvaluation,
)
from ferret import Benchmark

from app.explainers.model import XAIOutput


def get_evaluation(
    explanation: XAIOutput, model, tokenizer
) -> List[FerretExplanationEvaluation]:
    bench = Benchmark()

    # For each class in
