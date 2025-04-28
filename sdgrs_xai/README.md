# SDGRS XAI CLI toolkit

Works with: LLaMA and BERT family models.

## Features

* Produce explanations from different abstract sources (Qdrant DB, local dataset) and store the results (QDrant or locally). Multiprocessing and multi-gpu supported.
* Visualize explanations from a dataset for single examples with multiple XAI methods
* Evaluate faithfulness of explanations on a local dataset

## Quick start

```
python -m main --help
```

```
CUDA_VISIBLE_DEVICES=1,6 LOG_LEVEL=DEBUG python -m app.main --method=shap-partition-tfidf --model-family=scibert --model-path=/srv/scratch2/dbielik/.cache/huggingface/checkpoints/final/allenai/scibert_scivocab_cased-zo_up/checkpoint-432 --n-workers=5 --n-publications=384 --tfidf-corpus-path=./experiments/data/osdg.csv
```

```
CUDA_VISIBLE_DEVICES=0,1 LOG_LEVEL=INFO python -W ignore -m app.main --model-family=scibert --model-path=/srv/scratch2/dbielik/.cache/huggingface/checkpoints/final/allenai/scibert_scivocab_cased-zo_up/checkpoint-432 --data-source-target=qdrant --method=attnlrp explain --n-workers=5 --n-publications=384
```

```
CUDA_VISIBLE_DEVICES=0,1 LOG_LEVEL=INFO python -W ignore -m app.main --model-family=scibert --model-path=/srv/scratch2/dbielik/.cache/huggingface/checkpoints/final/allenai/scibert_scivocab_cased-zo_up/checkpoint-432 --data-source-target=local --input-path=./experiments/data/zo_up_test.csv --output-path=./heatmap_cli.pdf --method=shap-partition-tfidf --tfidf-corpus-path=experiments/data/osdg.csv visualize --index 12
```
