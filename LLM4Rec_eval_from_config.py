"""
Evaluate a saved stage-1 LLM4Rec checkpoint using the saved config.
"""

import json
import os

import fire
import torch

from model import LLM4Rec, load_and_preprocess_model_state_dict
from utils.data_utils import SequentialDataset
from utils.eval_utils import display_and_save_results, evaluate_model_improved
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
    updated_configs: str = "",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_eval_res: bool = False,
    eval_over_candidate_items: bool = False,
    save_all_user_eval_res: bool = False,
):
    set_seed(seed)
    save_eval_res = _normalize_bool(save_eval_res)
    eval_over_candidate_items = _normalize_bool(eval_over_candidate_items)
    save_all_user_eval_res = _normalize_bool(save_all_user_eval_res)

    print(f"Evaluating model from {load_model_path} ...")
    configs = json.load(open(os.path.join(load_model_path, "finetune_params.json"), "r"))
    if len(updated_configs) > 0:
        configs.update(json.loads(updated_configs))
    for key, value in configs.items():
        print(f"{key}: {value}")

    prompter = Prompter(configs["prompt_template_name"])
    dataset = SequentialDataset(configs["data_path"], 50)
    item_embed = torch.load(os.path.join(configs["data_path"], "SASRec_item_embed.pt"), map_location="cpu")

    model = LLM4Rec(
        base_model=configs["base_model"],
        task_type=configs["task_type"],
        cache_dir=configs["cache_dir"],
        input_dim=64,
        output_dim=dataset.m_item,
        lora_r=configs["lora_r"],
        lora_alpha=configs["lora_alpha"],
        lora_dropout=configs["lora_dropout"],
        lora_target_modules=configs["lora_target_modules"],
        device_map=device,
        instruction_text=prompter.generate_prompt(configs["task_type"]),
        input_embeds=item_embed,
    )

    checkpoint_path = os.path.join(load_model_path, "finetune_model.pt")
    model.load_state_dict(load_and_preprocess_model_state_dict(checkpoint_path), strict=False)
    print(f"Model loaded from {checkpoint_path}")
    model = model.to(device)
    torch.cuda.empty_cache()

    topk = [1, 5, 10, 20, 50]
    eval_res = evaluate_model_improved(
        model=model,
        dataset=dataset,
        eval_over_candidate_items=eval_over_candidate_items,
        topk=topk,
        return_all_user_eval_res=save_all_user_eval_res,
    )

    if save_all_user_eval_res:
        results, user_eval_res = eval_res
        user_eval_name = (
            "all_user_eval_results_over_candidate_items.csv"
            if eval_over_candidate_items
            else "all_user_eval_results_over_all_items.csv"
        )
        if save_eval_res:
            user_eval_res.to_csv(os.path.join(load_model_path, user_eval_name), index=False)
    else:
        results = eval_res

    all_item_results = results.get("all_item_ranking", results)
    display_and_save_results(
        results=all_item_results,
        topk=topk,
        filename=os.path.join(load_model_path, "test_results_over_all_items.txt") if save_eval_res else None,
    )
    if eval_over_candidate_items and "candidate_item_ranking" in results:
        display_and_save_results(
            results=results["candidate_item_ranking"],
            topk=topk,
            filename=os.path.join(load_model_path, "test_results_over_candidate_items.txt") if save_eval_res else None,
        )


if __name__ == "__main__":
    fire.Fire(evaluate)
