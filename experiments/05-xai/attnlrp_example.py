import os

# needs to be executed before importing torch or transformers
# server specific: 6, 7 on tars (48GB VRAM on 3090)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set the home directory for huggingface transformers (where the models are saved)
# by default this is '~/.cache/huggingface/hub'
# see https://stackoverflow.com/questions/61798573/where-does-hugging-faces-transformers-save-models
# server specific:
os.environ["HF_HOME"] = "/srv/scratch2/dbielik/.cache/huggingface"

import torch
import numpy as np
from pathlib import Path
from transformers import set_seed
from dotenv import load_dotenv

# increase the number of elements printed in the tensor
torch.set_printoptions(threshold=10_000)

# set the number of threads for BLAS libraries (for maxmizing reproducibility)
# warning: this will slow down the training
USE_DETERMINISTIC_ALGORITHMS = False
torch.use_deterministic_algorithms(USE_DETERMINISTIC_ALGORITHMS)
if USE_DETERMINISTIC_ALGORITHMS:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# check if CUDA is available
if not torch.cuda.is_available():
    print("Warning: CUDA not available!")

# path of the directory containing this file
BASE_DIR_PATH = Path.cwd().parent
# path of the data directory
DATA_DIR_PATH = BASE_DIR_PATH / "data"

HOME_DIR = os.path.expanduser("~")
CHECKPOINT_PATH = (
    os.getenv("HF_HOME") or HOME_DIR + "/.cache/huggingface"
) + "/checkpoints"

# set the seed for reproducibility
SEED = 1337
set_seed(SEED)

# Load .env file in the root of the repo
load_dotenv(BASE_DIR_PATH.parent / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")


#
import torch
from transformers import AutoTokenizer
from lxt.models.llama import LlamaForCausalLM, attnlrp
from lxt.utils import pdf_heatmap, clean_tokens

model = LlamaForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# apply AttnLRP rules
attnlrp.register(model)

prompt = """\
Context: Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall. As of November 2022, 310 people have died on Everest. Over 200 bodies remain on the mountain and have not been removed due to the dangerous conditions. The first recorded efforts to reach Everest's summit were made by British mountaineers. As Nepal did not allow foreigners to enter the country at the time, the British made several attempts on the north ridge route from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m (22,970 ft) on the North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), marking the first time a human had climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one of the greatest mysteries on Everest to this day: George Mallory and Andrew Irvine made a final summit attempt on 8 June but never returned, sparking debate as to whether they were the first to reach the top. Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, using the southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 Swiss expedition. The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first reported ascent of the peak from the north ridge on 25 May 1960. \
Question: How high did they climb in 1922? According to the text, the 1922 expedition reached 8,"""

input_ids = tokenizer(
    prompt, return_tensors="pt", add_special_tokens=True
).input_ids.to(model.device)
input_embeds = model.get_input_embeddings()(input_ids)

output_logits = model(
    inputs_embeds=input_embeds.requires_grad_(), use_cache=False
).logits
max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)

max_logits.backward(max_logits)
relevance = input_embeds.grad.float().sum(-1).cpu()[0]

# normalize relevance between [-1, 1] for plotting
relevance = relevance / relevance.abs().max()

# remove '_' characters from token strings
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print(tokens)
tokens = clean_tokens(tokens)
print(tokens)

pdf_heatmap(tokens, relevance, path="heatmap.pdf", backend="xelatex")
