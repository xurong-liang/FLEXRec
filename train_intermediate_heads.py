"""
Stage 2: initialize and train the intermediate prediction heads for LLM4Rec.
"""

import json
import os

import fire
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import transformers

from model import LLM4RecWithMultiPredHead, extract_incremental_state_dict, load_and_preprocess_model_state_dict
from utils.data_utils import SequentialCollator, SequentialDataset
from utils.eval_utils import (
    display_and_save_multihead_results_pretrain,
    evaluate_multihead_model_pretrain_optimized,
)
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


def runner(
    seed: int = 42,
    load_model_path: str = "",
    exit_layer_intervals: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_all_user_eval_res: bool = False,
    eval_over_candidate_items: bool = False,
    prompt_template_name: str = "alpaca",
    lr_scheduler: str = "cosine",
    additional_alias: str = None,
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    resume_from_checkpoint: str = None,
    warmup_steps: int = 100,
    learning_rate: float = 3e-4,
    val_set_size: int = 0,
    group_by_length: bool = False,
):
    del device
    resume_from_checkpoint = _normalize_optional_path(resume_from_checkpoint)
    save_all_user_eval_res = _normalize_bool(save_all_user_eval_res)
    eval_over_candidate_items = _normalize_bool(eval_over_candidate_items)
    group_by_length = _normalize_bool(group_by_length)
    pl.seed_everything(seed)
    if exit_layer_intervals != 1:
        raise ValueError("FLEXRec only supports exit_layer_intervals=1 in stage 2.")
    if batch_size % micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size.")
    if group_by_length:
        raise ValueError("group_by_length=True is not supported by FLEXRec SequentialDataset.")

    configs = json.load(open(os.path.join(load_model_path, "finetune_params.json"), "r"))
    output_folder_name = "LLM4RecWithMultiPredHead_exit_intervals_1"
    if additional_alias is not None:
        output_folder_name += f"_{additional_alias}"
    output_dir = os.path.join(load_model_path, output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    configs.update(
        {
            "output_dir": output_dir,
            "exit_layer_intervals": 1,
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "num_epochs": num_epochs,
            "resume_from_checkpoint": resume_from_checkpoint,
            "warmup_steps": warmup_steps,
            "learning_rate": learning_rate,
            "val_set_size": val_set_size,
            "group_by_length": group_by_length,
            "lr_scheduler": lr_scheduler,
        }
    )
    with open(os.path.join(output_dir, "train_intermediate_heads_params.json"), "w") as f:
        json.dump(configs, f, indent=4)

    gradient_accumulation_steps = batch_size // micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "cuda"

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps //= world_size

    dataset = SequentialDataset(configs["data_path"], 50)
    item_embed = torch.load(os.path.join(configs["data_path"], "SASRec_item_embed.pt"), map_location="cpu")
    data_collator = SequentialCollator()

    model = LLM4RecWithMultiPredHead(
        base_model=configs["base_model"],
        task_type=configs["task_type"],
        cache_dir=configs["cache_dir"],
        input_dim=64,
        output_dim=dataset.m_item,
        lora_r=configs["lora_r"],
        lora_alpha=configs["lora_alpha"],
        lora_dropout=configs["lora_dropout"],
        lora_target_modules=configs["lora_target_modules"],
        device_map=device_map,
        instruction_text=prompter.generate_prompt(configs["task_type"]),
        input_embeds=item_embed,
        exit_layer_intervals=1,
    )

    finetuned_model_path = os.path.join(load_model_path, "finetune_model.pt")
    model.load_state_dict(load_and_preprocess_model_state_dict(finetuned_model_path), strict=False)

    for name, param in model.named_parameters():
        param.requires_grad = "intermediate_heads" in name

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
            save_total_limit=2,
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
        model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        model_save_path = os.path.join(output_dir, "trained_intermediate_heads_model.pt")
        torch.save(extract_incremental_state_dict(model_to_save), model_save_path)

        del trainer
        del model
        torch.cuda.empty_cache()

        eval_model = LLM4RecWithMultiPredHead(
            base_model=configs["base_model"],
            task_type=configs["task_type"],
            cache_dir=configs["cache_dir"],
            input_dim=64,
            output_dim=dataset.m_item,
            lora_r=configs["lora_r"],
            lora_alpha=configs["lora_alpha"],
            lora_dropout=configs["lora_dropout"],
            lora_target_modules=configs["lora_target_modules"],
            device_map="cuda:0",
            instruction_text=prompter.generate_prompt(configs["task_type"]),
            input_embeds=item_embed,
            exit_layer_intervals=1,
        ).to("cuda:0")
        eval_model.load_state_dict(
            load_and_preprocess_model_state_dict(model_save_path),
            strict=False,
        )
        eval_model.eval()

        topk = [1, 5, 10, 20, 50]
        eval_res = evaluate_multihead_model_pretrain_optimized(
            eval_model,
            dataset,
            eval_over_candidate_items=eval_over_candidate_items,
            topk=topk,
            return_all_user_eval_res=save_all_user_eval_res,
        )

        if save_all_user_eval_res:
            results, user_eval_res = eval_res
            user_eval_dir = os.path.join(
                output_dir,
                "multihead_all_user_eval_results_over_candidate_items"
                if eval_over_candidate_items
                else "multihead_all_user_eval_results_over_all_items",
            )
            os.makedirs(user_eval_dir, exist_ok=True)
            for head_name, head_df in user_eval_res.items():
                head_df.to_csv(os.path.join(user_eval_dir, f"{head_name}.csv"), index=False)
        else:
            results = eval_res

        eval_filename = os.path.join(
            output_dir,
            "multihead_test_results_over_candidate_items.txt"
            if eval_over_candidate_items
            else "multihead_test_results_over_all_items.txt",
        )
        display_and_save_multihead_results_pretrain(results, topk, eval_filename)


if __name__ == "__main__":
    fire.Fire(runner)
