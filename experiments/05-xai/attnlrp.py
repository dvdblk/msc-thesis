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
    "[CLS]",
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

tokens_with_combined_words = [
    "[CLS]",
    "The",
    "quick",
    "brown",
    "fox",
    "jump",
    "##s",
    "over",
    "the",
    "lazy",
    "dog",
    "##s",
    ".",
    "[SEP]",
]

lime_tokens = [
    "Rein",
    "##sur",
    "##ance",
    "or",
    "Sec",
    "##uri",
    "##tization",
    ":",
    "The",
    "Case",
    "of",
    "Natural",
    "Cat",
    "##astro",
    "##ph",
    "##e",
    "Risk",
    "We",
    "investigate",
    "the",
    "suitability",
    "of",
    "sec",
    "##uri",
    "##tization",
    "as",
    "an",
    "alternative",
    "to",
    "rein",
    "##sur",
    "##ance",
    "for",
    "the",
    "purpose",
    "of",
    "transferring",
    "natural",
    "catast",
    "##roph",
    "##e",
    "risk",
    ".",
]


def fix_bert_tokenization(tokens):
    fixed_tokens = []
    for i, token in enumerate(tokens):
        if i == 0 and token == "[CLS]":
            fixed_tokens.append(token)
        elif i == len(tokens) - 1 and token == "[SEP]":
            fixed_tokens.append(token)
        elif token.startswith("##"):
            fixed_tokens.append(token[2:])
            if (
                i < len(tokens) - 1
                and tokens[i + 1] not in ["[SEP]", "", ",", ".", ":", ";", "!", "?"]
                and not tokens[i + 1].startswith("##")
            ):
                fixed_tokens[-1] += " "
        elif token in ["", ",", ".", ":", ";", "!", "?"]:
            if i > 0 and not tokens[i - 1] in [
                "[CLS]",
                "",
                ",",
                ".",
                ":",
                ";",
                "!",
                "?",
            ]:
                fixed_tokens.append(token + " ")
            else:
                fixed_tokens.append(token)
        elif (
            i < len(tokens) - 1
            and not tokens[i + 1].startswith("##")
            and tokens[i + 1] not in ["[SEP]", "", ",", ".", ":", ";", "!", "?"]
        ):
            fixed_tokens.append(token + " ")
        else:
            fixed_tokens.append(token)

    # Remove trailing space from the last token if it exists and it's not [SEP]
    if fixed_tokens and fixed_tokens[-1].endswith(" ") and fixed_tokens[-1] != "[SEP]":
        fixed_tokens[-1] = fixed_tokens[-1].rstrip()

    return fixed_tokens

print(fix_bert_tokenization(lime_tokens))

# assert fix_bert_tokenization(tokens_scibert_attnlrp) == tokens_scibert_shap, (
#     fix_bert_tokenization(tokens_scibert_attnlrp),
#     tokens_scibert_shap,
# )

relevance = torch.tensor([0, 1, 0])
# tokens = fix_bert_tokenization(tokens_scibert_attnlrp)
tokens = ["Hello ", "World", "!"]
assert len(tokens) == len(
    relevance
), f"Tokens ({len(tokens)}) and relevance scores ({len(relevance)}) must have the same length."
pdf_heatmap(
    tokens,
    relevance,
    path="heatmap_seq_cls.pdf",
    backend="xelatex",
)
