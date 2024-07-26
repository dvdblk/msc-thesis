from app.explainers import ExplainerMethod

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from lxt.models.bert import (
    BertForSequenceClassification as BertForSequenceClassificationLRP,
)
from lxt.models.llama import (
    LlamaForSequenceClassification as LlamaForSequenceClassificationLRP,
)

# TODO: scibert -> bert
__model_family_map = {
    "llama": None,
    "llama-unmasked": None,  # unmasked llama
    "scibert": None,
}


def get_model_names_list():
    """Returns a list of available model names"""
    return list(__model_family_map.keys())


def setup_model_and_tokenizer(args):
    # check if model family is valid
    if args.model_family not in __model_family_map:
        raise ValueError(f"Invalid model family: {args.model_family}")

    id2label = {i: str(i + 1) for i in range(17)}
    label2id = {str(i + 1): i for i in range(17)}

    method = ExplainerMethod(args.method)
    if method == ExplainerMethod.ATTNLRP or method == ExplainerMethod.CPLRP:
        # load model based on family
        if args.model_family == "scibert":
            model_class = BertForSequenceClassificationLRP
        elif args.model_family == "llama":
            model_class = LlamaForSequenceClassificationLRP
        else:
            raise ValueError(
                f"Unsupported model family for AttnLRP: {args.model_family}"
            )
    else:
        model_class = AutoModelForSequenceClassification

    # Load model
    model = model_class.from_pretrained(
        args.model_path,
        id2label=id2label,
        label2id=label2id,
    )
    model.eval()
    model.share_memory()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, do_lower_case=False)

    return model, tokenizer
