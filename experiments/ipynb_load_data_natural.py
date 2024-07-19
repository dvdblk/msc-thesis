"""Utility script for reuse within Jupyter Notebooks. Loads the common datasets and encodings.

Warning: requires ipynb_util.py to be executed first (before running this script).
"""

import torch
from enum import Enum
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import pickle

from ipynb_util_tars import DATA_DIR_PATH, BASE_DIR_PATH, SEED


class DatasetType(Enum):
    """Enum for the dataset type."""

    """Zora + OSDG upsampled dataset"""
    ZO_UP = "zo_up"
    """SwissText Shared Task 1 dataset (Zurich NLP)"""
    SWISSTEXT_SHARED_TASK1 = "swisstext_shared_task1"


TEST_SIZE = 0.3
DATASET_TYPE = DatasetType.ZO_UP

# load the dataset
# note: if you don't have the data in the folder, use the download-data.sh script

match DATASET_TYPE:
    case DatasetType.ZO_UP:
        # dont need to use manual features as class_encode_column will create ClassLabel
        # careful: watch out for the order of the ClassLabel as it doesn't map directly to the SDG class. need use mapping functions (id2label, label2id)
        # sdgs = [str(i) for i in range(1, 18)] + ["non-relevant"]
        # features = Features({"sdg": ClassLabel(num_classes=len(sdgs), names=sdgs), "abstract": Value("string")})

        dataset = load_dataset("csv", data_files=str(DATA_DIR_PATH / "zo_up.csv"))
        dataset = dataset.rename_columns({"sdg": "SDG", "abstract": "ABSTRACT"})

        def convert_sdg_to_0indexed_int(d):
            d["SDG"] = int(d["SDG"]) - 1
            return d

        dataset = dataset.map(convert_sdg_to_0indexed_int)

        train_df = dataset["train"].to_pandas()
        train_data, test_data = train_test_split(
            train_df, test_size=TEST_SIZE, stratify=train_df["SDG"], random_state=SEED
        )

        dataset["train"] = Dataset.from_pandas(train_data)
        dataset["test"] = Dataset.from_pandas(test_data)

        # dataset = dataset.class_encode_column("SDG")
        # dataset = dataset["train"].train_test_split(
        #    test_size=TEST_SIZE, stratify_by_column="SDG", seed=SEED
        # )
    case DatasetType.SWISSTEXT_SHARED_TASK1:
        dataset = load_dataset(
            "json",
            data_files=str(
                DATA_DIR_PATH / "swisstext-2024-sharedtask" / "task1-train.jsonl"
            ),
        )
        dataset = dataset["train"].train_test_split(test_size=TEST_SIZE, seed=SEED)

print(dataset["train"].features)
example = dataset["train"][0]
print("Example instance:\t", example)


# Label encodings / mappings
labels = set(dataset["train"]["SDG"])

# Create id2label and label2id dictionaries
id2label = {i: str(i + 1) for i in range(len(labels))}
label2id = {str(i + 1): i for i in range(len(labels))}

# Print the results
print("id2label:", id2label)
print("label2id:", label2id)

# save the encodings to a file for later use
ENCODING_DIR = BASE_DIR_PATH / "encodings" / DATASET_TYPE.value
# create the directory if it doesn't exist
ENCODING_DIR.mkdir(parents=True, exist_ok=True)
with open(ENCODING_DIR / "id2label_natural.pkl", "wb") as f:
    pickle.dump(id2label, f)

with open(ENCODING_DIR / "label2id_natural.pkl", "wb") as f:
    pickle.dump(label2id, f)

# verify that encodings work properly on example instance
# example instance has label 9 in the csv file
# example instance has label 16 in the encoded dataset
assert example["SDG"] == label2id[id2label[example["SDG"]]]
print("Encoded (label2id) label:\t", example["SDG"])
print("Decoded (id2label) label:\t", id2label[example["SDG"]])

print(id2label[16], label2id[id2label[16]], label2id["9"])

from transformers import AutoTokenizer

# whether the text should be lowered or not
SHOULD_LOWER = False


def preprocess_data(
    tokenizer, padding="max_length", max_length=512, include_labels=True
):
    def _preprocess_data(instances):
        match DATASET_TYPE:
            case DatasetType.SWISSTEXT_SHARED_TASK1:
                # take a batch of titles and abstracts and concat them
                titles = instances["TITLE"]
                abstracts = instances["ABSTRACT"]
                texts = [
                    f"{title} {abstract}" for title, abstract in zip(titles, abstracts)
                ]
            case DatasetType.ZO_UP:
                texts = instances["ABSTRACT"]

        if SHOULD_LOWER:
            texts = [text.lower() for text in texts]

        # encode
        encoding = tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if include_labels:
            # add labels
            encoding["label"] = torch.tensor([label for label in instances["SDG"]])

        return encoding

    return _preprocess_data
