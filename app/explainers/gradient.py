import torch
from captum.attr import InputXGradient, IntegratedGradients, Saliency
from torch.nn.parallel import DataParallel

from app.explainers.base import BaseExplainer
from app.explainers.model import XAIOutput, ExplainerMethod
from app.utils.tokenization import fix_bert_tokenization


class GradientExplainer(BaseExplainer):

    def __init__(self, model, tokenizer, device, max_seq_len, multiply_by_inputs=True):
        xai_method = ExplainerMethod.GRADIENTXINPUT
        super().__init__(model, tokenizer, device, max_seq_len, xai_method=xai_method)
        self.multiply_by_inputs = multiply_by_inputs

    def _predictor_func(self, input_embeds):
        outputs = self.model(inputs_embeds=input_embeds)
        return outputs.logits

    def explain(self, abstract):
        inputs = self.tokenizer(
            abstract,
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        input_embeds = self.model.get_input_embeddings()(inputs.input_ids)

        dl = (
            InputXGradient(self._predictor_func)
            if self.multiply_by_inputs
            else Saliency(self._predictor_func)
        )

        n_classes = self.model.num_labels
        input_len = inputs.attention_mask.sum().item()
        attr_all = torch.zeros((input_len, n_classes), device=self.device)

        for class_idx in range(n_classes):
            attr = dl.attribute(input_embeds, target=class_idx)
            attr = attr[0, :input_len, :].sum(-1)  # Pool over hidden size
            attr_all[:, class_idx] = attr

        logits = self._predictor_func(input_embeds)
        predicted_id = logits.argmax(dim=-1).item()
        predicted_label = self.model.config.id2label[predicted_id]
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()[0]

        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        tokens = fix_bert_tokenization(tokens)

        return XAIOutput(
            text=abstract,
            input_tokens=tokens,
            token_scores=attr_all.cpu().tolist(),
            predicted_id=predicted_id,
            predicted_label=predicted_label,
            probabilities=probabilities,
            xai_method=self.xai_method,
        )


class IntegratedGradientExplainer(BaseExplainer):

    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_seq_len,
        multiply_by_inputs=True,
        use_all_gpus=False,
    ):
        super().__init__(
            model,
            tokenizer,
            device,
            max_seq_len,
            xai_method=ExplainerMethod.INTEGRATED_GRADIENT,
        )
        self.multiply_by_inputs = multiply_by_inputs
        self.use_all_gpus = use_all_gpus

        if self.use_all_gpus and torch.cuda.device_count() > 1:
            self.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            self.parallel_model = DataParallel(model, device_ids=self.devices)
        else:
            self.devices = [device]
            self.parallel_model = model

    def _predictor_func(self, input_embeds):
        if self.use_all_gpus and torch.cuda.device_count() > 1:
            input_embeds = input_embeds.to(self.devices[0])
        outputs = self.parallel_model(inputs_embeds=input_embeds)
        return outputs.logits

    def _generate_baselines(self, input_len):
        # Account for all possible special tokens
        ids = [self.tokenizer.pad_token_id] * input_len
        if self.tokenizer.bos_token_id is not None:
            ids[0] = self.tokenizer.bos_token_id
        if self.tokenizer.eos_token_id is not None:
            ids[-1] = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is None:
            ids = [self.tokenizer.convert_tokens_to_ids([" "])[0]] * input_len

        embeddings = self.model.get_input_embeddings()(
            torch.tensor(ids, device=self.devices[0])
        )
        return embeddings.unsqueeze(0)

    def explain(self, abstract):
        inputs = self.tokenizer(
            abstract,
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.devices[0])
        input_embeds = self.model.get_input_embeddings()(inputs.input_ids)

        dl = IntegratedGradients(
            self._predictor_func, multiply_by_inputs=self.multiply_by_inputs
        )

        baselines = self._generate_baselines(inputs.attention_mask.sum().item())

        n_classes = self.model.num_labels

        # Compute attributions for each class separately
        attr_all = []
        for class_idx in range(n_classes):
            attr = dl.attribute(
                input_embeds,
                baselines=baselines,
                target=class_idx,
                n_steps=50,
                internal_batch_size=2,
            )
            attr = attr.sum(-1)  # Pool over hidden size
            attr_all.append(attr)

        # Stack the attributions for all classes
        attr_all = torch.stack(attr_all, dim=1)

        logits = self._predictor_func(input_embeds)
        predicted_id = logits.argmax(dim=-1).item()
        predicted_label = self.model.config.id2label[predicted_id]
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()[0]

        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        tokens = fix_bert_tokenization(tokens)

        # get to the correct shape expected by the visualize method ...
        attr_all = attr_all.squeeze(0).transpose(0, 1)

        return XAIOutput(
            text=abstract,
            input_tokens=tokens,
            token_scores=attr_all.cpu().tolist(),
            predicted_id=predicted_id,
            predicted_label=predicted_label,
            probabilities=probabilities,
            xai_method=self.xai_method,
        )
