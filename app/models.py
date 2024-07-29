import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lxt.models.bert import (
    BertForSequenceClassification as BertForSequenceClassificationLRP,
)
from lxt.models.llama import (
    LlamaForSequenceClassification as LlamaForSequenceClassificationLRP,
)

from app.explainers import ExplainerMethod

# TODO: scibert -> bert
__model_family_map = {
    "llama2": None,  # also for TinyLlama
    "llama3": None,
    "llama3-unmasked": None,  # unmasked llama
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
    model = None
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, do_lower_case=False)

    if method == ExplainerMethod.ATTNLRP or method == ExplainerMethod.CPLRP:
        # load model based on family
        if args.model_family == "scibert":
            model = BertForSequenceClassificationLRP.from_pretrained(
                args.model_path, id2label=id2label, label2id=label2id
            )
        elif args.model_family == "llama3" or args.model_family == "llama2":
            model = LlamaForSequenceClassificationLRP.from_pretrained(
                args.model_path,
                id2label=id2label,
                label2id=label2id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            model.config.pad_token_id = tokenizer.pad_token_id
        else:
            raise ValueError(
                f"Unsupported model family for AttnLRP: {args.model_family}"
            )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            id2label=id2label,
            label2id=label2id,
            torch_dtype=(
                torch.bfloat16
                if args.model_family == "llama3" or args.model_family == "llama2"
                else torch.float32
            ),
        )
        if args.model_family == "llama3" or args.model_family == "llama2":
            model.config.pad_token_id = tokenizer.pad_token_id

    # Load model
    model.eval()
    model.share_memory()

    return model, tokenizer
