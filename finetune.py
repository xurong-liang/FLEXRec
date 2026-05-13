import json
import os
from typing import List

import fire
import torch
import torch.distributed as dist
import transformers

from model import LLM4Rec, extract_incremental_state_dict
from utils.data_utils import SequentialCollator, SequentialDataset
from utils.eval_utils import display_and_save_results, evaluate_model_improved
from utils.prompter import Prompter


def _normalize_optional_path(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def _normalize_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n", ""}:
            return False
    return value


def train(
    base_model: str = "",
    data_path: str = "",
    cache_dir: str = "",
    output_dir: str = "",
    task_type: str = "sequential",
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    val_set_size: int = 0,
    lr_scheduler: str = "cosine",
    warmup_steps: int = 100,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["gate_proj", "down_proj", "up_proj"],
    train_on_inputs: bool = False,
    add_eos_token: bool = False,
    group_by_length: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
    prompt_template_name: str = "alpaca",
    additional_alias: str = None,
    eval_over_candidate_items: bool = False,
):
    resume_from_checkpoint = _normalize_optional_path(resume_from_checkpoint)
    train_on_inputs = _normalize_bool(train_on_inputs)
    add_eos_token = _normalize_bool(add_eos_token)
    group_by_length = _normalize_bool(group_by_length)
    eval_over_candidate_items = _normalize_bool(eval_over_candidate_items)

    if task_type != "sequential":
        raise ValueError("FLEXRec finetune.py only supports task_type='sequential'.")
    if not base_model:
        raise ValueError("Please specify --base_model.")
    if batch_size % micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size.")
    if group_by_length:
        raise ValueError("group_by_length=True is not supported by FLEXRec SequentialDataset.")

    dataset_name = os.path.basename(os.path.normpath(data_path))
    output_folder = f"{base_model.split('/')[-1]}-{task_type}-{num_epochs:d}-epochs"
    if additional_alias is not None:
        output_folder += f"-{additional_alias}"
    output_dir = os.path.join(output_dir, dataset_name, output_folder)
    os.makedirs(output_dir, exist_ok=True)

    kwargs = {
        "base_model": base_model,
        "data_path": data_path,
        "cache_dir": cache_dir,
        "output_dir": output_dir,
        "task_type": task_type,
        "batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "cutoff_len": cutoff_len,
        "val_set_size": val_set_size,
        "lr_scheduler": lr_scheduler,
        "warmup_steps": warmup_steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": lora_target_modules,
        "train_on_inputs": train_on_inputs,
        "add_eos_token": add_eos_token,
        "group_by_length": group_by_length,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "wandb_watch": wandb_watch,
        "wandb_log_model": wandb_log_model,
        "resume_from_checkpoint": resume_from_checkpoint,
        "prompt_template_name": prompt_template_name,
        "eval_over_candidate_items": eval_over_candidate_items,
    }
    with open(os.path.join(output_dir, "finetune_params.json"), "w") as f:
        json.dump(kwargs, f, indent=4)

    gradient_accumulation_steps = batch_size // micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "cuda"

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps //= world_size

    dataset = SequentialDataset(data_path, 50)
    item_embed = torch.load(os.path.join(data_path, "SASRec_item_embed.pt"), map_location="cpu")
    data_collator = SequentialCollator()

    model = LLM4Rec(
        base_model=base_model,
        task_type=task_type,
        cache_dir=cache_dir,
        input_dim=64,
        output_dim=dataset.m_item,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        device_map=device_map,
        instruction_text=prompter.generate_prompt(task_type),
        input_embeds=item_embed,
    )

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer_checkpoint_dir = os.path.join(output_dir, "transformers_checkpoint")
    os.makedirs(trainer_checkpoint_dir, exist_ok=True)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=None,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=1000,
            lr_scheduler_type=lr_scheduler,
            output_dir=trainer_checkpoint_dir,
            save_total_limit=1,
            load_best_model_at_end=val_set_size > 0,
            ddp_find_unused_parameters=False,
            group_by_length=group_by_length,
            report_to="none",
            run_name=None,
        ),
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if ddp:
        dist.barrier()
    if (ddp and dist.get_rank() == 0) or not ddp:
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(extract_incremental_state_dict(model_to_save), os.path.join(output_dir, "finetune_model.pt"))

        topk = [1, 5, 10, 20, 50]
        results = evaluate_model_improved(
            model=model_to_save,
            dataset=dataset,
            eval_over_candidate_items=eval_over_candidate_items,
            topk=topk,
        )
        all_item_results = results.get("all_item_ranking", results)
        display_and_save_results(
            all_item_results,
            topk,
            os.path.join(output_dir, "test_results_over_all_items.txt"),
        )
        if eval_over_candidate_items and "candidate_item_ranking" in results:
            display_and_save_results(
                results["candidate_item_ranking"],
                topk,
                os.path.join(output_dir, "test_results_over_candidate_items.txt"),
            )


if __name__ == "__main__":
    fire.Fire(train)
