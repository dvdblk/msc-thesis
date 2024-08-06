"""Evaluation script for all LLMs."""

import torch.nn as nn
import torch
from transformers.models.llama.modeling_llama import (
    LlamaForSequenceClassification,
    LlamaDecoderLayer,
    LlamaConfig,
    LlamaRMSNorm,
    LlamaModel,
    LLAMA_INPUTS_DOCSTRING,
    add_start_docstrings_to_model_forward,
    BaseModelOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache
from typing import Optional, List, Union, Tuple


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from sklearn.metrics import classification_report


set_seed(1337)
# list of tuples (model_type, model_path)
__models = [
    (
        "unllama3",
        "/final/meta-llama/Meta-Llama-3-8B-ft-zo_up-unmasked/checkpoint-2528/",
    ),
    ("llama2", "/TinyLlama/TinyLlama_v1.1/checkpoint-80"),
    ("scibert", "/final/allenai/scibert_scivocab_cased-zo_up/checkpoint-432/"),
    ("llama2", "/final/meta-llama/Llama-2-7b-hf/checkpoint-1264/"),
    ("llama3", "/final/meta-llama/Meta-Llama-3-8B-ft-zo_up/checkpoint-2212/"),
]

MODEL_BASE_PATH = "/srv/scratch2/dbielik/.cache/huggingface/checkpoints"
DATA_DIR_PATH = "/home/user/dbielik/msc-thesis/experiments/data"


class UnmaskingLlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(
            past_key_values, Cache
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )
        if causal_mask is not None:
            # print("b4", input_ids.shape, causal_mask.shape, causal_mask)
            # Assuming causal_mask is a tensor with shape (batch_size, 1, seq_length, hidden_size)
            causal_mask_last_row = causal_mask[:, :, -1, :].unsqueeze(2)
            causal_mask = causal_mask_last_row.expand_as(causal_mask)
            # causal_mask = torch.zeros_like(causal_mask, device=inputs_embeds.device)

            # print("after", causal_mask.shape, causal_mask)
        else:
            pass
            # print("kek it's none", causal_mask, input_ids)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class UnmaskingLlamaForSequenceClassification(LlamaForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = UnmaskingLlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


def preprocess_data(
    tokenizer, padding="max_length", max_length=512, include_labels=True
):
    def _preprocess_data(instances):
        texts = instances["ABSTRACT"]

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


if __name__ == "__main__":
    for model_type, model_path in __models:
        # Load the test set
        dataset = load_dataset(
            "csv", data_files={"test": f"{DATA_DIR_PATH}/zo_up_test.csv"}
        )
        dataset = dataset.rename_columns({"sdg": "SDG", "abstract": "ABSTRACT"})

        def convert_sdg_to_0indexed_int(d):
            d["SDG"] = int(d["SDG"]) - 1
            return d

        dataset = dataset.map(convert_sdg_to_0indexed_int)
        # Label encodings / mappings
        labels = set(dataset["test"]["SDG"])

        # Create id2label and label2id dictionaries
        id2label = {i: str(i + 1) for i in range(len(labels))}
        label2id = {str(i + 1): i for i in range(len(labels))}

        print(f"Model: {model_path}")
        model_path = MODEL_BASE_PATH + model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
        if model_type == "unllama3":
            model = UnmaskingLlamaForSequenceClassification.from_pretrained(
                model_path,
                label2id=label2id,
                id2label=id2label,
                device_map="auto",
            ).bfloat16()
            model.config.pad_token_id = tokenizer.pad_token_id
        elif model_type == "llama2" or model_type == "llama3":
            model = LlamaForSequenceClassification.from_pretrained(
                model_path, label2id=label2id, id2label=id2label, device_map="auto"
            ).bfloat16()
            model.config.pad_token_id = tokenizer.pad_token_id
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, label2id=label2id, id2label=id2label
            ).to("cuda")
        model.eval()

        # preprocess the data
        encoded_dataset = dataset.map(
            preprocess_data(tokenizer, max_length=512, padding=True),
            batched=True,
            remove_columns=dataset["test"].column_names,
        )
        encoded_dataset.set_format("torch")

        # manual evaluation to show classifcation_report
        true_labels = []
        logits = []

        for batch in encoded_dataset["test"]:
            batch = {k: v.to(model.device).unsqueeze(0) for k, v in batch.items()}
            label = batch.pop("label")

            # Forward pass
            with torch.no_grad():
                out = model(**batch)

            true_labels.append(label.item())
            logits.extend(out.logits.tolist())

        probabilites = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        pred_labels = torch.argmax(probabilites, dim=-1).tolist()

        report = classification_report(
            true_labels,
            pred_labels,
            target_names=[f"SDG {id2label[i]}" for i in range(len(labels))],
            digits=4,
        )
        print(report)
        print()
        torch.cuda.empty_cache()
