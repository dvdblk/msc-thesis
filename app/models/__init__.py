from functools import wraps
import torch

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lxt.models.bert import (
    BertForSequenceClassification as BertForSequenceClassificationLRP,
)
from lxt.models.llama import (
    LlamaForSequenceClassification as LlamaForSequenceClassificationLRP,
)
from lxt.models.unllama import (
    UnmaskingLlamaForSequenceClassification as UnmaskingLlamaForSequenceClassificationLRP,
)

from app.explainers import ExplainerMethod
from app.models.unllama import UnmaskingLlamaForSequenceClassification

# TODO: scibert -> bert
__model_family_map = {
    "llama2": None,  # also works for TinyLlama
    "llama3": None,
    "unllama3": None,  # unmasked llama
    "scibert": None,
}


def get_model_names_list():
    """Returns a list of available model names"""
    return list(__model_family_map.keys())


def enforce_max_length(func):
    """Force tokenizer to have 512 seq length"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs["truncation"] = True
        kwargs["max_length"] = args[0].model_max_length
        return func(*args, **kwargs)

    return wrapper


def setup_model_and_tokenizer(args):
    # check if model family is valid
    if args.model_family not in __model_family_map:
        raise ValueError(f"Invalid model family: {args.model_family}")

    id2label = {i: str(i + 1) for i in range(17)}
    label2id = {str(i + 1): i for i in range(17)}

    method = ExplainerMethod(args.method)
    model = None
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, do_lower_case=False)

    # TODO: as with explainer.py, this should be model dependent
    # tokenizer.model_max_length = 512
    # tokenizer.__call__ = enforce_max_length(tokenizer.__call__)

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
                torch_dtype=torch.bfloat16,
            )
            model.config.pad_token_id = tokenizer.pad_token_id

            # remove lora weights for LRP to work
            peft_model = PeftModel.from_pretrained(model, model_id=args.model_path)
            model = peft_model.merge_and_unload()
        elif args.model_family == "unllama3":
            # FIXME: replace with LRP capable Unllama3
            model = UnmaskingLlamaForSequenceClassificationLRP.from_pretrained(
                args.model_path,
                id2label=id2label,
                label2id=label2id,
                torch_dtype=torch.bfloat16,
            )
            model.config.pad_token_id = tokenizer.pad_token_id
            # remove lora weights for LRP to work
            peft_model = PeftModel.from_pretrained(model, model_id=args.model_path)
            model = peft_model.merge_and_unload()
        else:
            raise ValueError(
                f"Unsupported model family for AttnLRP: {args.model_family}"
            )
    else:
        if args.model_family == "unllama3":
            model = UnmaskingLlamaForSequenceClassification.from_pretrained(
                args.model_path,
                id2label=id2label,
                label2id=label2id,
                torch_dtype=torch.bfloat16,
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
        if (
            args.model_family == "llama3"
            or args.model_family == "llama2"
            or args.model_family == "unllama3"
        ):
            model.config.pad_token_id = tokenizer.pad_token_id

    # Load model
    model.eval()
    model.share_memory()

    return model, tokenizer
