"""
Evaluate a saved stage-2 LLM4RecWithMultiPredHead checkpoint using the saved config.
"""

import json
import os

import fire
import torch

from model import LLM4RecWithMultiPredHead, load_and_preprocess_model_state_dict
from utils.data_utils import SequentialDataset
from utils.eval_utils import (
    display_and_save_multihead_results_pretrain,
    evaluate_multihead_model_pretrain_optimized,
)
from utils.general_utils import set_seed
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

    print(f"Evaluating model from {load_model_path} ...")
    configs = json.load(open(os.path.join(load_model_path, "train_intermediate_heads_params.json"), "r"))
    configs["output_dir"] = load_model_path
    for key, value in configs.items():
        print(f"{key}: {value}")

    dataset = SequentialDataset(configs["data_path"], 50)
    item_embed = torch.load(os.path.join(configs["data_path"], "SASRec_item_embed.pt"), map_location="cpu")
    prompter = Prompter(prompt_template_name)

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
        exit_layer_intervals=configs["exit_layer_intervals"],
    ).to("cuda:0")

    model_save_path = os.path.join(load_model_path, "trained_intermediate_heads_model.pt")
    eval_model.load_state_dict(load_and_preprocess_model_state_dict(model_save_path), strict=False)
    eval_model.eval()
    print("Model reloaded successfully for evaluation")

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
        if save_eval_res:
            user_eval_dir = os.path.join(
                load_model_path,
                "multihead_all_user_eval_results_over_candidate_items"
                if eval_over_candidate_items
                else "multihead_all_user_eval_results_over_all_items",
            )
            os.makedirs(user_eval_dir, exist_ok=True)
            for key, df in user_eval_res.items():
                df.to_csv(os.path.join(user_eval_dir, f"{key}.csv"), index=False)
    else:
        results = eval_res

    eval_filename = None
    if save_eval_res:
        eval_filename = os.path.join(
            load_model_path,
            "multihead_test_results_over_candidate_items.txt"
            if eval_over_candidate_items
            else "multihead_test_results_over_all_items.txt",
        )
    display_and_save_multihead_results_pretrain(results, topk, eval_filename)


if __name__ == "__main__":
    fire.Fire(evaluate)
