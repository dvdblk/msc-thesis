import os

import numpy as np
from ferret import Benchmark
from ferret.explainers.explanation import Explanation as FerretExplanation
from ferret.evaluators.evaluation import (
    ExplanationEvaluation as FerretExplanationEvaluation,
)


class XAIEvaluator:
    """
    A class for evaluating XAI explanations for faithfulness using the Ferret library.
    """

    def __init__(self, model, tokenizer, device):
        """
        Initialize the XAIEvaluator.

        Args:
            model: The model to be evaluated.
            tokenizer: The tokenizer used by the model.
            device: The device (CPU or GPU) to run the evaluation on.
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.bench = Benchmark(model, tokenizer)

    def evaluate_explanation(
        self, xai_output, class_index
    ) -> FerretExplanationEvaluation:
        """
        Evaluate a single explanation for a specific class.

        Args:
            xai_output: The output from the XAI method.
            class_index: The index of the class to evaluate.

        Returns:
            An evaluation object from the Ferret library.
        """
        _tokenized_text = self.tokenizer.tokenize(
            xai_output.text,
            max_length=len(xai_output.input_tokens),
            add_special_tokens=True,
        )

        _token_ids = self.tokenizer.convert_tokens_to_ids(_tokenized_text)
        decoded_text_from_tokens = self.tokenizer.decode(_token_ids)

        ferret_explanation = FerretExplanation(
            text=decoded_text_from_tokens,
            tokens=_tokenized_text,
            scores=np.array(xai_output.token_scores)[:, class_index],
            explainer=xai_output.xai_method.value,
            target=class_index,
        )

        evaluation = self.bench.evaluate_explanation(
            ferret_explanation,
            class_index,
            show_progress=False,
            remove_first_last=False,
        )

        return evaluation

    def evaluate_sample(self, xai_output) -> np.ndarray:
        """
        Evaluate all classes for a single sample.

        Args:
            xai_output: The output from the XAI method for a single sample.

        Returns:
            A numpy array of shape (num_labels, 3) containing evaluation scores for each class.
            Scores in the order: aopc_compr, aopc_suff, taucorr_loo
        """
        evaluations_for_explanations = np.zeros((self.model.num_labels, 3))

        for class_idx in range(self.model.num_labels):
            evaluation_for_class = self.evaluate_explanation(xai_output, class_idx)
            evaluations_for_explanations[class_idx] = np.array(
                list(map(lambda e: e.score, evaluation_for_class.evaluation_scores))
            )

        return evaluations_for_explanations

    def save_evaluation(
        self, evaluation_array, model_name, xai_method, base_dir="evaluations"
    ):
        # Create directory if it doesn't exist
        dir_path = os.path.join(base_dir, model_name)
        os.makedirs(dir_path, exist_ok=True)

        # Generate file name
        file_name = f"{xai_method}_evaluation.npy"
        file_path = os.path.join(dir_path, file_name)

        # Save the numpy array
        np.save(file_path, evaluation_array)

    def load_evaluation(self, model_name, xai_method, base_dir="evaluations"):
        # Generate file name
        file_name = f"{xai_method}_evaluation.npy"
        file_path = os.path.join(base_dir, model_name, file_name)

        # Load the numpy array
        return np.load(file_path)
