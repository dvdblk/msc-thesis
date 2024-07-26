import torch

from app.explainers.lrp import AttnLRPExplainer, CPLRPExplainer
from app.explainers.shap import (
    KernelShapExplainer,
    PartitionShapExplainer,
    TfIdfPartitionShapExplainer,
)

from enum import StrEnum, auto


class ExplainerMethod(StrEnum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower().replace("_", "-")

    SHAP_PARTITION = auto()
    SHAP_KERNEL = auto()
    SHAP_PARTITION_TFIDF = auto()
    ATTNLRP = auto()
    CPLRP = auto()
    LIME = auto()
    INTEGRATED_GRADIENTS = auto()
    INPUTXGRADIENT = auto()


# Map of XAI methods to their respective Explainer classes
__methods_map = {
    ExplainerMethod.SHAP_PARTITION: PartitionShapExplainer,
    ExplainerMethod.SHAP_KERNEL: KernelShapExplainer,
    ExplainerMethod.SHAP_PARTITION_TFIDF: TfIdfPartitionShapExplainer,
    ExplainerMethod.ATTNLRP: AttnLRPExplainer,
    ExplainerMethod.CPLRP: CPLRPExplainer,
    ExplainerMethod.LIME: None,
    ExplainerMethod.INTEGRATED_GRADIENTS: None,
    ExplainerMethod.INPUTXGRADIENT: None,
}


def get_xai_method_names_list():
    """Returns a list of available XAI method names"""
    # only list the methods that have an associated explainer class
    methods_with_explainer = [k for k, v in __methods_map.items() if v is not None]
    return methods_with_explainer


def process_abstract(abstract, explainer, model, tokenizer, device):
    # Get prediction
    inputs = tokenizer(
        [abstract], padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_id = torch.argmax(probs, dim=-1).item()
    predicted_label = model.config.id2label[predicted_id]

    # Get explanation (apply XAI method)
    xai_values = explainer(abstract)

    return predicted_label, probs.tolist(), xai_values


def get_explainer(method_name, model, tokenizer, device, args):
    """Returns an instance of the specified XAI method"""
    # verify that the method is supported
    if method_name not in __methods_map:
        raise ValueError(f"Unsupported XAI method: {method_name}")

    ExplainerClass = __methods_map[method_name]
    method = ExplainerMethod(method_name)

    if method == ExplainerMethod.SHAP_PARTITION_TFIDF:
        return ExplainerClass(args.tfidf_corpus_path, model, tokenizer, device)
    elif method == ExplainerMethod.ATTNLRP:
        return ExplainerClass(args.model_family, model, tokenizer, device)
    else:
        return ExplainerClass(model, tokenizer, device)
