# XAI thesis code

## Quickstart

1. Install `venv`: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
2. (Optional step) Install `cupy` for a [correct CUDA version](https://docs.cupy.dev/en/stable/install.html#requirements): e.g. `pip install cupy-cuda12x==12.3.0`
3. `xelatex` for AttnLRP: `conda install -c conda-forge texlive-core`

## Experiments

* `01-scibert` = reproduced models from jroady thesis (including transformers and not only SpaCy)
* `02-llama` = xai on ft llama models
* `03-hf-repr` = reproduce models in huggingface (SCIBERT, LLaMA-2, LLaMA-3)

### Misc.

* [how-to commit ipynb without outputs](https://gist.github.com/33eyes/431e3d432f73371509d176d0dfb95b6e?permalink_comment_id=4662892)