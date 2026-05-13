import os
from collections import OrderedDict

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaConfig,
    LlamaModel,
    Qwen3Config,
    Qwen3Model,
)
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


INCREMENTAL_STATE_KEY_PATTERNS = (
    "input_proj.",
    "score.",
    "intermediate_heads.",
    "moe.",
)


def _is_incremental_parameter_name(name: str) -> bool:
    if "lora_" in name:
        return True
    return any(name.startswith(prefix) for prefix in INCREMENTAL_STATE_KEY_PATTERNS)


def _hf_auth_kwargs() -> dict:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return {"token": hf_token} if hf_token else {}


class PerLayerPropLlamaConfig(LlamaConfig):
    model_type = "per_layer_prop_llama"


class PerLayerPropLlamaModelForMLP(LlamaModel):
    config_class = PerLayerPropLlamaConfig

    def __init__(self, config):
        if not isinstance(config, PerLayerPropLlamaConfig):
            config = PerLayerPropLlamaConfig(**config.to_dict())
        super().__init__(config)

    def forward_single_layer(
        self,
        layer_idx,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        output_attentions=False,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = False
        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if layer_idx == 0:
            causal_mask = create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=None,
            )
        else:
            causal_mask = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        decoder_layer = self.layers[layer_idx]
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = layer_outputs[0]
        hidden_states_with_norm = self.norm(hidden_states)
        attention_weights = layer_outputs[1] if output_attentions else None
        present_key_value = layer_outputs[2 if output_attentions else 1] if use_cache else None
        return {
            "hidden_states_wo_norm": hidden_states,
            "hidden_states_with_norm": hidden_states_with_norm,
            "causal_mask": causal_mask,
            "position_ids": position_ids,
            "attention_weights": attention_weights,
            "present_key_value": present_key_value,
            "layer_idx": layer_idx,
            "is_last_layer": layer_idx == len(self.layers) - 1,
        }


AutoConfig.register("per_layer_prop_llama", PerLayerPropLlamaConfig)
AutoModel.register(PerLayerPropLlamaConfig, PerLayerPropLlamaModelForMLP)


class PerLayerPropQwen3Config(Qwen3Config):
    model_type = "per_layer_prop_qwen3"


class PerLayerPropQwen3ModelForMLP(Qwen3Model):
    config_class = PerLayerPropQwen3Config

    def __init__(self, config):
        if not isinstance(config, PerLayerPropQwen3Config):
            config = PerLayerPropQwen3Config(**config.to_dict())
        super().__init__(config)

    def forward_single_layer(
        self,
        layer_idx,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        output_attentions=False,
        **kwargs,
    ):
        use_cache = False
        hidden_states = inputs_embeds
        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if layer_idx == 0:
            causal_mask = create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=None,
            )
        else:
            causal_mask = attention_mask

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        decoder_layer = self.layers[layer_idx]
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = layer_outputs[0]
        hidden_states_with_norm = self.norm(hidden_states)
        attention_weights = layer_outputs[1] if output_attentions else None
        present_key_value = layer_outputs[2 if output_attentions else 1] if use_cache else None
        return {
            "hidden_states_wo_norm": hidden_states,
            "hidden_states_with_norm": hidden_states_with_norm,
            "causal_mask": causal_mask,
            "position_ids": position_ids,
            "attention_weights": attention_weights,
            "present_key_value": present_key_value,
            "layer_idx": layer_idx,
            "is_last_layer": layer_idx == len(self.layers) - 1,
        }


AutoConfig.register("per_layer_prop_qwen3", PerLayerPropQwen3Config)
AutoModel.register(PerLayerPropQwen3Config, PerLayerPropQwen3ModelForMLP)


class LLM4Rec(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.args = args
        self.input_dim = args["input_dim"]
        self.output_dim = args["output_dim"]
        if args["task_type"] != "sequential":
            raise ValueError("FLEXRec only supports task_type='sequential'.")

        print("Initializing language decoder ...")
        peft_config = LoraConfig(
            task_type="FEATURE_EXTRACTION",
            r=self.args["lora_r"],
            lora_alpha=self.args["lora_alpha"],
            lora_dropout=self.args["lora_dropout"],
            target_modules=self.args["lora_target_modules"],
            bias="none",
        )
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        hf_auth_kwargs = _hf_auth_kwargs()

        if "Llama" in self.args["base_model"]:
            self.llm_model = LlamaModel.from_pretrained(
                self.args["base_model"],
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                cache_dir=args["cache_dir"],
                device_map=self.args["device_map"],
                **hf_auth_kwargs,
            )
        elif "Qwen3" in self.args["base_model"]:
            self.llm_model = Qwen3Model.from_pretrained(
                self.args["base_model"],
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                cache_dir=args["cache_dir"],
                device_map=self.args["device_map"],
                **hf_auth_kwargs,
            )
        else:
            self.llm_model = AutoModel.from_pretrained(
                self.args["base_model"],
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                cache_dir=args["cache_dir"],
                device_map=self.args["device_map"],
                **hf_auth_kwargs,
            )

        self.llm_model = prepare_model_for_kbit_training(self.llm_model)
        self.llm_model = get_peft_model(self.llm_model, peft_config)
        self.llm_model.print_trainable_parameters()
        self.llm_model.config.use_cache = False

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.args["base_model"],
            use_fast=False,
            cache_dir=args["cache_dir"],
            **hf_auth_kwargs,
        )
        self.llm_tokenizer.pad_token_id = 0
        self.llm_tokenizer.pad_token = "[PAD]"
        self.llm_tokenizer.padding_side = "right"
        self.instruct_ids, self.instruct_mask = self.llm_tokenizer(
            self.args["instruction_text"][0],
            truncation=True,
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
        ).values()
        self.response_ids, self.response_mask = self.llm_tokenizer(
            self.args["instruction_text"][1],
            truncation=True,
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
        ).values()
        print("Language decoder initialized.")

        self.task_type = args["task_type"]
        self.input_embeds = nn.Embedding.from_pretrained(self.args["input_embeds"], freeze=True)
        self.input_proj = nn.Linear(self.input_dim, self.llm_model.config.hidden_size)
        self.score = nn.Linear(self.llm_model.config.hidden_size, self.output_dim, bias=False)

    def predict(self, inputs, inputs_mask, output_hidden_states: bool = False):
        bs = inputs.shape[0]
        if isinstance(self.llm_model, PeftModel):
            instruct_embeds = self.llm_model.model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
            response_embeds = self.llm_model.model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        else:
            instruct_embeds = self.llm_model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
            response_embeds = self.llm_model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        inputs = self.input_proj(self.input_embeds(inputs))
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        outputs = self.llm_model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )
        pooled_output = outputs.last_hidden_state[:, -1]
        pooled_logits = self.score(pooled_output)
        return outputs, pooled_logits.view(-1, self.output_dim)

    def forward(self, inputs, inputs_mask, labels, **kwargs):
        del kwargs
        outputs, pooled_logits = self.predict(inputs, inputs_mask)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(pooled_logits, labels.view(-1))
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LLM4RecWithMultiPredHead(LLM4Rec):
    def __init__(self, exit_layer_intervals: int = 4, **kwargs):
        super().__init__(**kwargs)
        if self.llm_model.config.num_hidden_layers % exit_layer_intervals != 0:
            raise ValueError(
                f"Number of hidden layers {self.llm_model.config.num_hidden_layers} "
                f"must be divisible by exit_interval {exit_layer_intervals}."
            )

        self.exit_interval = exit_layer_intervals
        self.exit_layer_idxes = list(range(0, self.llm_model.config.num_hidden_layers, exit_layer_intervals))[1:]
        self.num_heads = self.llm_model.config.num_hidden_layers // exit_layer_intervals
        self.intermediate_heads = nn.ModuleList(
            [nn.Linear(self.llm_model.config.hidden_size, self.output_dim, bias=False) for _ in range(self.num_heads - 1)]
        )

        del self.llm_model
        print(f"Initializing language decoder with {self.exit_interval} exit intervals ...")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        hf_auth_kwargs = _hf_auth_kwargs()
        if "Llama" in self.args["base_model"]:
            self.llm_model = PerLayerPropLlamaModelForMLP.from_pretrained(
                self.args["base_model"],
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                cache_dir=kwargs["cache_dir"],
                device_map=self.args["device_map"],
                **hf_auth_kwargs,
            )
        elif "Qwen3" in self.args["base_model"]:
            self.llm_model = PerLayerPropQwen3ModelForMLP.from_pretrained(
                self.args["base_model"],
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                cache_dir=kwargs["cache_dir"],
                device_map=self.args["device_map"],
                **hf_auth_kwargs,
            )
        else:
            raise NotImplementedError("Only Llama and Qwen3 models are supported in this version.")

        self.llm_model = prepare_model_for_kbit_training(self.llm_model)
        peft_config = LoraConfig(
            task_type="FEATURE_EXTRACTION",
            r=self.args["lora_r"],
            lora_alpha=self.args["lora_alpha"],
            lora_dropout=self.args["lora_dropout"],
            target_modules=self.args["lora_target_modules"],
            bias="none",
        )
        self.llm_model = get_peft_model(self.llm_model, peft_config)
        self.llm_model.print_trainable_parameters()
        self.llm_model.config.use_cache = False
        print(f"Exit layer indexes: {self.exit_layer_idxes}")

    def predict(
        self,
        inputs,
        inputs_mask,
        output_hidden_states: bool = False,
        return_inference_time_by_head: bool = False,
    ) -> dict:
        head_time_events = None
        head_time_values_sec = None
        time_start = None
        time_marks_sec = None
        if return_inference_time_by_head:
            if torch.is_tensor(inputs) and inputs.is_cuda and torch.cuda.is_available():
                head_time_events = []
                time_start = torch.cuda.Event(enable_timing=True)
                time_start.record()
            else:
                import time as _time

                time_start = _time.perf_counter()
                time_marks_sec = []

        bs = inputs.shape[0]
        if isinstance(self.llm_model, PeftModel):
            instruct_embeds = self.llm_model.model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
            response_embeds = self.llm_model.model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        else:
            instruct_embeds = self.llm_model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
            response_embeds = self.llm_model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        inputs = self.input_proj(self.input_embeds(inputs))
        input_embeds = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        all_hidden_states = () if output_hidden_states else None
        all_pred_head_outputs = ()

        outputs = self.llm_model.forward_single_layer(0, input_embeds, attention_mask, position_ids=None)
        if output_hidden_states:
            all_hidden_states += (outputs["hidden_states_wo_norm"],)

        if self.exit_interval == 1:
            inter_logits = self.intermediate_heads[len(all_pred_head_outputs)](outputs["hidden_states_with_norm"][:, -1])
            all_pred_head_outputs += (inter_logits.view(-1, self.output_dim),)
            if return_inference_time_by_head:
                if head_time_events is not None:
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    head_time_events.append(ev)
                else:
                    import time as _time

                    time_marks_sec.append(_time.perf_counter() - time_start)

        for layer_idx in range(1, self.llm_model.config.num_hidden_layers):
            outputs = self.llm_model.forward_single_layer(
                layer_idx,
                outputs["hidden_states_wo_norm"],
                attention_mask=outputs["causal_mask"],
                position_ids=outputs["position_ids"],
            )
            if output_hidden_states:
                all_hidden_states += (outputs["hidden_states_wo_norm"],)

            if layer_idx + 1 in self.exit_layer_idxes:
                if len(all_pred_head_outputs) < len(self.intermediate_heads):
                    inter_logits = self.intermediate_heads[len(all_pred_head_outputs)](
                        outputs["hidden_states_with_norm"][:, -1]
                    )
                    all_pred_head_outputs += (inter_logits.view(-1, self.output_dim),)
                    if return_inference_time_by_head:
                        if head_time_events is not None:
                            ev = torch.cuda.Event(enable_timing=True)
                            ev.record()
                            head_time_events.append(ev)
                        else:
                            import time as _time

                            time_marks_sec.append(_time.perf_counter() - time_start)

        all_pred_head_outputs += (self.score(outputs["hidden_states_with_norm"][:, -1]).view(-1, self.output_dim),)
        if return_inference_time_by_head:
            if head_time_events is not None:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                head_time_events.append(ev)
            else:
                import time as _time

                time_marks_sec.append(_time.perf_counter() - time_start)

        if return_inference_time_by_head:
            if head_time_events is not None:
                torch.cuda.synchronize(inputs.device)
                previous_event = time_start
                head_time_values_sec = []
                for event in head_time_events:
                    head_time_values_sec.append(previous_event.elapsed_time(event) / 1000.0)
                    previous_event = event
            else:
                head_time_values_sec = list(time_marks_sec)
        return {
            "all_hidden_states": all_hidden_states,
            "all_pred_head_outputs": all_pred_head_outputs,
            "final_hidden_states": outputs["hidden_states_with_norm"],
            "inference_time_by_head_sec": head_time_values_sec,
            "inference_time_start_event": time_start if head_time_events is not None else None,
            "inference_time_by_head_events": head_time_events,
        }

    def forward(self, inputs, inputs_mask, labels, add_final_head_loss: bool = False, **kwargs):
        del kwargs
        outputs = self.predict(inputs, inputs_mask, output_hidden_states=False)
        all_pred_head_outputs = outputs["all_pred_head_outputs"]
        all_hidden_states = outputs["all_hidden_states"]

        if labels is not None:
            losses = []
            for logits in all_pred_head_outputs[:-1]:
                losses.append(nn.CrossEntropyLoss()(logits, labels.view(-1)))
            if add_final_head_loss:
                losses.append(nn.CrossEntropyLoss()(all_pred_head_outputs[-1], labels.view(-1)))
        total_loss = torch.stack(losses).mean() if labels is not None else None
        return SequenceClassifierOutputWithPast(
            loss=total_loss,
            logits=torch.stack(all_pred_head_outputs),
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


def load_and_preprocess_model_state_dict(finetuned_path: str) -> OrderedDict:
    checkpoint = torch.load(finetuned_path, map_location="cpu")
    if any(key.startswith("module.") for key in checkpoint.keys()):
        new_checkpoint = OrderedDict()
        for key, value in checkpoint.items():
            new_checkpoint[key[7:] if key.startswith("module.") else key] = value
        checkpoint = new_checkpoint

    if any("llama_model" in key for key in checkpoint.keys()):
        new_checkpoint = OrderedDict()
        for key, value in checkpoint.items():
            new_checkpoint[key.replace("llama_model.", "llm_model.") if key.startswith("llama_model.") else key] = value
        checkpoint = new_checkpoint
    return checkpoint


def extract_incremental_state_dict(model) -> OrderedDict:
    model_to_save = model.module if hasattr(model, "module") else model
    incremental_state = OrderedDict()
    for name, tensor in model_to_save.state_dict().items():
        if _is_incremental_parameter_name(name):
            incremental_state[name] = tensor.detach().cpu().clone()
    return incremental_state
