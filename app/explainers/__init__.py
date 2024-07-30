import torch

from app.explainers.lrp import AttnLRPExplainer, CPLRPExplainer
from app.explainers.shap import (
    KernelShapExplainer,
    PartitionShapExplainer,
    TfIdfPartitionShapExplainer,
)
from app.explainers.gradient import (
    GradientExplainer,
    IntegratedGradientExplainer,
)
from app.explainers.lime import LimeExplainer
from app.explainers.model import ExplainerMethod


# Map of XAI methods to their respective Explainer classes
__methods_map = {
    ExplainerMethod.SHAP_PARTITION: PartitionShapExplainer,
    ExplainerMethod.SHAP_KERNEL: KernelShapExplainer,
    ExplainerMethod.SHAP_PARTITION_TFIDF: TfIdfPartitionShapExplainer,
    ExplainerMethod.ATTNLRP: AttnLRPExplainer,
    ExplainerMethod.CPLRP: CPLRPExplainer,
    ExplainerMethod.LIME: LimeExplainer,
    ExplainerMethod.INTEGRATED_GRADIENT: IntegratedGradientExplainer,
    ExplainerMethod.GRADIENTXINPUT: GradientExplainer,
}


def get_xai_method_names_list():
    """Returns a list of available XAI method names"""
    # only list the methods that have an associated explainer class
    methods_with_explainer = [k for k, v in __methods_map.items() if v is not None]
    return methods_with_explainer


def get_explainer(method_name, model, tokenizer, device, args):
    """Returns an instance of the specified XAI method"""
    # verify that the method is supported
    if method_name not in __methods_map:
        raise ValueError(f"Unsupported XAI method: {method_name}")

    ExplainerClass = __methods_map[method_name]
    method = ExplainerMethod(method_name)
    # FIXME: change to 1024 with bigger VRAM or add DataParallel
    max_seq_len = 512 if args.model_family == "scibert" else 512

    if method == ExplainerMethod.SHAP_PARTITION_TFIDF:
        # pad the end of TFIDF mask in case the model is 'scibert'
        should_pad_end_of_mask = args.model_family == "scibert"

        return ExplainerClass(
            args.tfidf_corpus_path,
            should_pad_end_of_mask,
            model,
            tokenizer,
            device,
            max_seq_len,
        )
    elif method == ExplainerMethod.ATTNLRP or method == ExplainerMethod.CPLRP:
        return ExplainerClass(args.model_family, model, tokenizer, device, max_seq_len)
    else:
        return ExplainerClass(model, tokenizer, device, max_seq_len)
