import torch
from lxt.utils import pdf_heatmap

# from TinyLLama (attnlrp), preserve tokens!
tokens = [
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
tokens = [
    "",
    "An ",
    "example ",
    "is ",
    "the ",
    "Eastern ",
    "Sc",
    "oti",
    "an ",
    "Shelf ",
    "ecosystem ",
    "that ",
]
relevance = torch.tensor([0, 0.1, 0.2, 0, 0.1, 0.3, 0.8, 0.1, 0, 0, 0, 0.4])
pdf_heatmap(
    tokens,
    relevance,
    path="heatmap_seq_cls.pdf",
    backend="xelatex",
)
