import os

# needs to be executed before importing torch or transformers
# server specific: only use last 3 gpus (on rattle.ifi.uzh.ch)
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"
# set the home directory for huggingface transformers (where the models are saved)
# by default this is '~/.cache/huggingface/hub'
# see https://stackoverflow.com/questions/61798573/where-does-hugging-faces-transformers-save-models
# server specific:
# os.environ["HF_HOME"] = "/srv/scratch2/dbielik/.cache/huggingface"

import torch
import numpy as np
from pathlib import Path
from transformers import set_seed

torch.set_printoptions(threshold=10_000)

USE_DETERMINISTIC_ALGORITHMS = False
torch.use_deterministic_algorithms(USE_DETERMINISTIC_ALGORITHMS)
if USE_DETERMINISTIC_ALGORITHMS:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

if not torch.cuda.is_available():
    print("Warning: CUDA not available!")

# path of the directory containing this file
BASE_DIR_PATH = Path.cwd().parent
# path of the data directory
DATA_DIR_PATH = BASE_DIR_PATH / "data" / "swisstext-2024-sharedtask"

CHECKPOINT_PATH = (os.getenv("HF_HOME") or "..") + "/checkpoints"

SEED = 1337
set_seed(SEED)
