import json
import os
from datetime import timedelta
from timeit import default_timer as timer

import fire
import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.optimization import get_scheduler

from MOE_model import FLEXRec
from model import load_and_preprocess_model_state_dict
from utils.data_utils import SequentialCollator, SequentialDataset
from utils.eval_utils import display_and_save_results, evaluate_model_improved
from utils.general_utils import print_text, set_seed
from utils.prompter import Prompter

torch.autograd.set_detect_anomaly(True)
EVAL_TOPK = [1, 5, 10, 20, 50]
TRAINABLE_COMPONENTS = ["moe"]


def _all_item_results(eval_results):
    if isinstance(eval_results, dict):
        return eval_results.get("all_item_ranking", eval_results)
    return eval_results


def print_moe_evaluation_insights(moe_stats: dict, file_fp=None):
    if not moe_stats:
        print_text("MOE HEAD USAGE ANALYSIS", file_fp)
        print_text("No evaluable users were found for this split.", file_fp)
        return
    print_text("MOE HEAD USAGE ANALYSIS", file_fp)
    print_text("\n1. OVERALL HEAD USAGE FREQUENCY:", file_fp)
    print_text("-" * 35, file_fp)
    for head_idx in sorted(moe_stats["head_usage_percentage"].keys()):
        count = moe_stats["head_usage_count"][head_idx]
        percentage = moe_stats["head_usage_percentage"][head_idx]
        avg_weight = moe_stats["head_avg_weight"][head_idx]
        print_text(
            f"  Head {head_idx}: {count:4d} users ({percentage:5.1f}%) | Avg Weight: {avg_weight:.3f}",
            file_fp,
        )

    print_text("\n2. LOAD BALANCING:", file_fp)
    print_text("-" * 20, file_fp)
    print_text(
        "  Overall Active Heads per User:",
        file_fp,
    )
    print_text(
        f"    Avg: {moe_stats['avg_active_heads_per_user']:.2f} | "
        f"Std: {moe_stats['std_active_heads_per_user']:.2f} | "
        f"Range: [{moe_stats['min_active_heads_per_user']:.0f}, {moe_stats['max_active_heads_per_user']:.0f}]",
        file_fp,
    )

    print_text("\n3. MOST COMMON HEAD COMBINATIONS:", file_fp)
    print_text("-" * 40, file_fp)
    for i, (combination, count) in enumerate(moe_stats["most_common_head_combinations"][:5]):
        total = sum(c for _, c in moe_stats["most_common_head_combinations"])
        percentage = count / total * 100 if total > 0 else 0.0
        heads_str = ", ".join(map(str, combination))
        print_text(f"  {i+1}. Heads [{heads_str}]: {count} users ({percentage:.1f}%)", file_fp)


def save_flexrec_network(model, save_path: str):
    save_state_dict = {}
    for name, param in model.named_parameters():
        clean_name = name.replace("module.", "") if name.startswith("module.") else name
        if any(component in clean_name for component in TRAINABLE_COMPONENTS):
            save_state_dict[clean_name] = param.cpu().detach().clone()
    torch.save(save_state_dict, save_path)


def print_debug_info(epoch, batch_idx, model, avg_epoch_losses, outputs, file_fp=None):
    exit_weight_mat = outputs["exit_head_weight_mat"]
    gate_inputs = outputs["gate_inputs"]
    with torch.no_grad():
        w_gate = model.moe.w_gate
        logits = gate_inputs @ w_gate
        nonzero_mask = exit_weight_mat > 0
        num_nonzero = nonzero_mask.sum().item()
        num_total = exit_weight_mat.numel()
        sparsity_ratio = (1 - num_nonzero / num_total) * 100
        active_heads_per_sample = nonzero_mask.sum(dim=1).float()
        avg_active_heads = active_heads_per_sample.mean().item()

    print_text("\n" + "=" * 80, file_fp)
    print_text(f"DEBUG INFO - FLEXRec - Epoch {epoch:2d} | Batch {batch_idx+1:4d}", file_fp)
    print_text("=" * 80, file_fp)
    print_text("ROUTING STATISTICS:", file_fp)
    print_text("-" * 40, file_fp)
    print_text(
        f"  Logits         -> Mean: {logits.mean().item():7.4f} | "
        f"Range: [{logits.min().item():6.3f}, {logits.max().item():6.3f}]",
        file_fp,
    )
    print_text(
        f"  Sparsity       -> {num_nonzero:4d}/{num_total:4d} nonzero ({sparsity_ratio:5.1f}% sparse)",
        file_fp,
    )
    print_text(
        f"  Active Heads   -> Avg per sample: {avg_active_heads:.2f} | "
        f"Range: [{active_heads_per_sample.min().item():.0f}, {active_heads_per_sample.max().item():.0f}]",
        file_fp,
    )
    print_text("\nTRAINING LOSSES:", file_fp)
    print_text("-" * 40, file_fp)
    for loss_name, loss_value in avg_epoch_losses.items():
        print_text(f"  {loss_name:20s}: {loss_value:8.4f}", file_fp)


def final_evaluation(
    model,
    eval_over_candidate_items,
    output_folder,
    load_path,
    train_dataset,
    save_all_user_eval_res: bool = False,
    file_fp=None,
):
    print_text("\n" + "*" * 100, file_fp)
    print_text(
        "Evaluating over all items and candidate items..."
        if eval_over_candidate_items
        else "Evaluating over all items...",
        file_fp,
    )
    policy_state_dict = load_and_preprocess_model_state_dict(load_path)
    model.load_state_dict(policy_state_dict, strict=False)
    print_text(f"Model loaded from {load_path}", file_fp)

    eval_res = evaluate_model_improved(
        model=model,
        dataset=train_dataset,
        eval_over_candidate_items=eval_over_candidate_items,
        topk=EVAL_TOPK,
        return_all_user_eval_res=save_all_user_eval_res,
        partition="test",
    )
    if save_all_user_eval_res:
        results, user_eval_res = eval_res
        user_eval_name = (
            "all_user_eval_results_over_candidate_items.csv"
            if eval_over_candidate_items
            else "all_user_eval_results_over_all_items.csv"
        )
        user_eval_res.to_csv(os.path.join(output_folder, user_eval_name), index=False)
    else:
        results = eval_res

    all_item_results = _all_item_results(results)
    print_text("\n=== FLEXRec HEAD USAGE ANALYSIS ===", file_fp)
    print_moe_evaluation_insights(all_item_results.get("moe_head_stats"), file_fp)

    all_item_results_to_save = dict(all_item_results)
    all_item_results_to_save.pop("moe_head_stats", None)
    display_and_save_results(
        all_item_results_to_save,
        EVAL_TOPK,
        os.path.join(output_folder, "test_results_over_all_items.txt"),
    )
    if eval_over_candidate_items and "candidate_item_ranking" in results:
        candidate_results = dict(results["candidate_item_ranking"])
        candidate_results.pop("moe_head_stats", None)
        display_and_save_results(
            candidate_results,
            EVAL_TOPK,
            os.path.join(output_folder, "test_results_over_candidate_items.txt"),
        )


def train_flexrec(
    model,
    optimizer,
    scheduler,
    dataset,
    train_loader,
    accelerator,
    configs,
    save_path,
    batch_eval_steps: int = 500,
    sampled_valid_user_idxes: list = None,
    debug_print_steps: int = 500,
    file_fp=None,
):
    best_ndcg_5 = -float("inf")
    patience = 0

    if accelerator.is_main_process:
        valid_res = evaluate_model_improved(
            model=model,
            dataset=dataset,
            eval_over_candidate_items=configs["eval_over_candidate_items"],
            partition="valid",
            sampled_user_idxes=sampled_valid_user_idxes,
        )
        valid_all_item_res = _all_item_results(valid_res)
        print_moe_evaluation_insights(valid_all_item_res.get("moe_head_stats"), file_fp)

    global_step = 0
    for epoch in range(configs["num_epochs"]):
        model.train()
        epoch_loss_sums = {
            "total_loss": 0.0,
            "pred_loss": 0.0,
            "target_k_loss": 0.0,
            "lb_loss": 0.0,
            "z_loss": 0.0,
        }
        num_batches = 0
        progress_bar = None
        if accelerator.is_main_process:
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{configs['num_epochs']}",
                total=len(train_loader),
                dynamic_ncols=True,
            )
            batch_iterator = enumerate(progress_bar)
        else:
            batch_iterator = enumerate(train_loader)

        for batch_idx, batch in batch_iterator:
            with accelerator.accumulate(model):
                outputs = model(
                    batch["inputs"],
                    batch["inputs_mask"],
                    batch["inputs"],
                    batch["labels"],
                )
                loss = outputs["total_loss"]
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            for key in epoch_loss_sums:
                epoch_loss_sums[key] += outputs[key].detach().float().item()
            num_batches += 1
            global_step += 1

            if progress_bar is not None:
                progress_bar.set_postfix(
                    loss=f"{outputs['total_loss'].detach().float().item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            if accelerator.is_main_process and global_step % debug_print_steps == 0:
                avg_epoch_losses = {k: v / num_batches for k, v in epoch_loss_sums.items()}
                unwrapped_model = accelerator.unwrap_model(model)
                print_debug_info(epoch, batch_idx, unwrapped_model, avg_epoch_losses, outputs, file_fp)

            if accelerator.is_main_process and batch_eval_steps > 0 and global_step % batch_eval_steps == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                valid_res = evaluate_model_improved(
                    model=unwrapped_model,
                    dataset=dataset,
                    eval_over_candidate_items=configs["eval_over_candidate_items"],
                    partition="valid",
                    sampled_user_idxes=sampled_valid_user_idxes,
                )
                valid_all_item_res = _all_item_results(valid_res)
                current_ndcg_5 = valid_all_item_res["NDCG"][1]
                print_moe_evaluation_insights(valid_all_item_res.get("moe_head_stats"), file_fp)
                print_text(f"Validation NDCG@5: {current_ndcg_5:.4f}", file_fp)

                if current_ndcg_5 > best_ndcg_5:
                    best_ndcg_5 = current_ndcg_5
                    patience = 0
                    save_flexrec_network(unwrapped_model, save_path)
                    print_text(f"Saved best FLEXRec checkpoint to {save_path}", file_fp)
                else:
                    patience += 1

                model.train()

                if configs["early_stop_patience"] > 0 and patience >= configs["early_stop_patience"]:
                    if progress_bar is not None:
                        progress_bar.close()
                    print_text("Early stopping triggered.", file_fp)
                    return

        if progress_bar is not None:
            progress_bar.close()

        if accelerator.is_main_process:
            avg_epoch_losses = {k: v / max(num_batches, 1) for k, v in epoch_loss_sums.items()}
            print_text(f"\nEpoch {epoch + 1}/{configs['num_epochs']} summary", file_fp)
            for key, value in avg_epoch_losses.items():
                print_text(f"  {key}: {value:.4f}", file_fp)

    if accelerator.is_main_process and not os.path.exists(save_path):
        save_flexrec_network(accelerator.unwrap_model(model), save_path)


def runner(
    seed: int = 42,
    load_model_path: str = "",
    save_all_user_eval_res: bool = False,
    eval_over_candidate_items: bool = False,
    prompt_template_name: str = "alpaca",
    additional_alias: str = None,
    batch_size: int = 128,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    val_set_size: int = -1,
    group_by_length: bool = False,
    eval_steps: int = 500,
    debug_print_steps: int = 250,
    warmup_steps: int = 100,
    save_stdout_to_file: bool = True,
    target_k: float = 3.0,
    tau: float = 10.0,
    _lambda: float = 1.,
    alpha: float = 0.05,
    beta: float = 1e-3,
    gamma: float = 2.0,
    pred_loss_weight: float = 1.0,
    eps: float = 0.1,
    early_stop_patience: int = -1,
):
    set_seed(seed)
    base_configs = json.load(open(os.path.join(load_model_path, "train_intermediate_heads_params.json"), "r"))
    if base_configs.get("exit_layer_intervals") != 1:
        raise ValueError("FLEXRec stage 3 only supports stage-2 checkpoints trained with exit_layer_intervals=1.")
    if base_configs.get("task_type") != "sequential":
        raise ValueError("FLEXRec stage 3 only supports task_type='sequential'.")

    output_folder_name = (
        f"FLEXRec_target_k_{target_k:.1f}_alpha_{alpha:.4f}_lambda_{_lambda:.2f}"
        f"_gamma_{gamma:.2f}_beta_{beta:.2e}_tau_{tau:.1f}"
    )
    output_folder_name += f"_eps_{eps:.4e}"
    if pred_loss_weight != 1.0:
        output_folder_name += f"_pred_{pred_loss_weight:.4f}"
    if additional_alias is not None:
        output_folder_name += f"_{additional_alias}"

    output_folder = os.path.join(load_model_path, output_folder_name)
    os.makedirs(output_folder, exist_ok=False)

    configs = {
        **base_configs,
        "output_dir": output_folder,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "group_by_length": group_by_length,
        "save_all_user_eval_res": save_all_user_eval_res,
        "eval_over_candidate_items": eval_over_candidate_items,
        "eval_steps": eval_steps,
        "val_set_size": val_set_size,
        "early_stop_patience": early_stop_patience,
        "pred_loss_weight": pred_loss_weight,
        "target_k": target_k,
        "tau": tau,
        "_lambda": _lambda,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "eps": eps,
    }
    with open(os.path.join(output_folder, "train_FLEXRec_params.json"), "w") as f:
        json.dump(configs, f, indent=4)

    prompter = Prompter(prompt_template_name)
    train_dataset = SequentialDataset(configs["data_path"], 50)
    if val_set_size != -1:
        sampled_users_for_valid = np.random.choice(
            sorted(list(train_dataset.valData.keys())),
            val_set_size,
            replace=False,
        )
    else:
        sampled_users_for_valid = None

    item_embed = torch.load(os.path.join(configs["data_path"], "SASRec_item_embed.pt"), map_location="cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=True,
        collate_fn=SequentialCollator(),
    )
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16", project_dir=output_folder)

    model = FLEXRec(
        base_model=configs["base_model"],
        task_type=configs["task_type"],
        cache_dir=configs["cache_dir"],
        input_dim=64,
        output_dim=train_dataset.m_item,
        lora_r=configs["lora_r"],
        lora_alpha=configs["lora_alpha"],
        lora_dropout=configs["lora_dropout"],
        lora_target_modules=configs["lora_target_modules"],
        device_map="cuda",
        instruction_text=prompter.generate_prompt(configs["task_type"]),
        input_embeds=item_embed,
        exit_layer_intervals=configs["exit_layer_intervals"],
        target_k=target_k,
        tau=tau,
        _lambda=_lambda,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        pred_loss_weight=pred_loss_weight,
        eps=eps,
    )
    intermediate_heads_model_path = os.path.join(load_model_path, "trained_intermediate_heads_model.pt")
    model.load_state_dict(load_and_preprocess_model_state_dict(intermediate_heads_model_path), strict=False)

    for name, param in model.named_parameters():
        param.requires_grad = any(component in name for component in TRAINABLE_COMPONENTS)

    optimizer = AdamW(model.parameters(), lr=configs["learning_rate"])
    num_training_steps = len(train_loader) * configs["num_epochs"]
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    model, optimizer, scheduler, train_loader = accelerator.prepare(model, optimizer, scheduler, train_loader)
    save_path = os.path.join(configs["output_dir"], "trained_FLEXRec.pt")
    file_fp = (
        open(os.path.join(configs["output_dir"], "train_FLEXRec_stdout.txt"), "w")
        if save_stdout_to_file and accelerator.is_main_process
        else None
    )

    if accelerator.is_main_process:
        process_start = timer()

    train_flexrec(
        model,
        optimizer,
        scheduler,
        train_dataset,
        train_loader,
        accelerator,
        configs,
        save_path,
        batch_eval_steps=configs["eval_steps"],
        sampled_valid_user_idxes=sampled_users_for_valid,
        debug_print_steps=debug_print_steps,
        file_fp=file_fp,
    )

    if accelerator.is_main_process:
        final_evaluation(
            model=accelerator.unwrap_model(model),
            eval_over_candidate_items=configs["eval_over_candidate_items"],
            output_folder=configs["output_dir"],
            load_path=save_path,
            train_dataset=train_dataset,
            save_all_user_eval_res=save_all_user_eval_res,
            file_fp=file_fp,
        )
        print_text(f"\nTotal time elapsed: {timedelta(seconds=int(timer() - process_start))}", file_fp)
        if file_fp is not None:
            file_fp.close()


if __name__ == "__main__":
    fire.Fire(runner)
