"""
Evaluate a saved stage-3 FLEXRec checkpoint using the saved config.
"""

import json
import os
from datetime import timedelta
from timeit import default_timer as timer

import fire
import torch

from MOE_model import FLEXRec
from model import load_and_preprocess_model_state_dict
from train_FLEXRec import final_evaluation
from utils.data_utils import SequentialDataset
from utils.general_utils import print_text, set_seed
from utils.prompter import Prompter


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


def evaluate(
    seed: int = 42,
    load_model_path: str = "",
    prompt_template_name: str = "alpaca",
    save_all_user_eval_res: bool = False,
    eval_over_candidate_items: bool = False,
    save_eval_res: bool = False,
):
    set_seed(seed)
    save_all_user_eval_res = _normalize_bool(save_all_user_eval_res)
    eval_over_candidate_items = _normalize_bool(eval_over_candidate_items)
    save_eval_res = _normalize_bool(save_eval_res)

    print(f"Evaluating FLEXRec model from {load_model_path} ...")
    configs = json.load(open(os.path.join(load_model_path, "train_FLEXRec_params.json"), "r"))
    configs["output_dir"] = load_model_path
    configs["eval_over_candidate_items"] = eval_over_candidate_items
    for key, value in configs.items():
        print(f"{key}: {value}")

    dataset = SequentialDataset(configs["data_path"], 50)
    item_embed = torch.load(os.path.join(configs["data_path"], "SASRec_item_embed.pt"), map_location="cpu")
    prompter = Prompter(prompt_template_name)

    eval_model = FLEXRec(
        base_model=configs["base_model"],
        task_type=configs["task_type"],
        cache_dir=configs["cache_dir"],
        input_dim=64,
        output_dim=dataset.m_item,
        lora_r=configs["lora_r"],
        lora_alpha=configs["lora_alpha"],
        lora_dropout=configs["lora_dropout"],
        lora_target_modules=configs["lora_target_modules"],
        device_map="cuda",
        instruction_text=prompter.generate_prompt(configs["task_type"]),
        input_embeds=item_embed,
        exit_layer_intervals=configs["exit_layer_intervals"],
        target_k=configs["target_k"],
        tau=configs["tau"],
        _lambda=configs["_lambda"],
        alpha=configs["alpha"],
        beta=configs["beta"],
        gamma=configs["gamma"],
        pred_loss_weight=configs.get("pred_loss_weight", 1.0),
        eps=configs.get("eps", 0.1),
    ).cuda()

    parent_dir = os.path.dirname(load_model_path.rstrip("/"))
    stage2_model_path = os.path.join(parent_dir, "trained_intermediate_heads_model.pt")
    eval_model.load_state_dict(load_and_preprocess_model_state_dict(stage2_model_path), strict=False)
    print(f"Loaded stage-2 weights from {stage2_model_path}")

    load_path = os.path.join(load_model_path, "trained_FLEXRec.pt")
    file_fp = None
    if save_eval_res:
        stdout_filename = (
            "eval_only_FLEXRec_stdout_over_all_and_candidate_items.txt"
            if eval_over_candidate_items
            else "eval_only_FLEXRec_stdout_over_all_items.txt"
        )
        file_fp = open(os.path.join(load_model_path, stdout_filename), "w")

    eval_start = timer()
    final_evaluation(
        model=eval_model,
        eval_over_candidate_items=eval_over_candidate_items,
        output_folder=load_model_path,
        load_path=load_path,
        train_dataset=dataset,
        save_all_user_eval_res=save_all_user_eval_res,
        file_fp=file_fp,
    )
    print_text(f"Total evaluation time: {timedelta(seconds=timer() - eval_start)}", file_fp)
    if file_fp is not None:
        file_fp.close()


if __name__ == "__main__":
    fire.Fire(evaluate)
