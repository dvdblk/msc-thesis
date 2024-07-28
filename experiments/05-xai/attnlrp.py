import torch
from lxt.utils import pdf_heatmap

# from TinyLLama (attnlrp), preserve tokens!
tokens_tinyllama_attnlrp = [
    "<s>",
    "▁Context",
    ":",
    "▁Mount",
    "▁Ever",
    "est",
    "▁attract",
    "s",
    "▁many",
    "▁clim",
    "bers",
    ",",
]
# from SciBERT (shap)
tokens_scibert_shap = [
    "",
    "The ",
    "future ",
    "of ",
    "online ",
    "content ",
    "personal",
    "isation",
    ": ",
    "Technology",
    ", ",
    "law ",
]

tokens_scibert_attnlrp = [
    "",
    "The",
    "future",
    "of",
    "online",
    "content",
    "personal",
    "##isation",
    ":",
    "Technology",
    ",",
    "law",
]


def fix_bert_tokenization(tokens):
    fixed_tokens = []
    for i, token in enumerate(tokens):
        if token.startswith("##"):
            fixed_tokens.append(token[2:])
        elif token in ["", ",", ".", ":", ";", "!", "?"]:
            if i > 0 and not tokens[i - 1] in ["", ",", ".", ":", ";", "!", "?"]:
                fixed_tokens.append(token + " ")
            else:
                fixed_tokens.append(token)
        elif (
            i < len(tokens) - 1
            and not tokens[i + 1].startswith("##")
            and tokens[i + 1] not in ["", ",", ".", ":", ";", "!", "?"]
        ):
            fixed_tokens.append(token + " ")
        else:
            fixed_tokens.append(token)

    # Remove trailing space from the last token if it exists
    if fixed_tokens and fixed_tokens[-1].endswith(" "):
        fixed_tokens[-1] = fixed_tokens[-1].rstrip()

    return fixed_tokens


assert fix_bert_tokenization(tokens_scibert_attnlrp) == tokens_scibert_shap, (
    fix_bert_tokenization(tokens_scibert_attnlrp),
    tokens_scibert_shap,
)

relevance = torch.tensor([0, 0.1, 0.2, 0, 0.1, 0.3, 0.8, 0.1, 0, 0, 0, 0.4])
tokens = fix_bert_tokenization(tokens_scibert_attnlrp)
assert len(tokens) == len(
    relevance
), f"Tokens ({len(tokens)}) and relevance scores ({len(relevance)}) must have the same length."
pdf_heatmap(
    tokens,
    relevance,
    path="heatmap_seq_cls.pdf",
    backend="xelatex",
)
