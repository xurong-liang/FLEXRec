import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def _is_metric_vector(val, expected_len: int) -> bool:
    if isinstance(val, (list, tuple, np.ndarray)):
        try:
            return len(val) == expected_len
        except TypeError:
            return False
    return False


def _format_inference_summary(avg_user_inference_time_sec, users_per_second):
    if avg_user_inference_time_sec is None and users_per_second is None:
        return None
    avg = 0.0 if avg_user_inference_time_sec is None else float(avg_user_inference_time_sec)
    ups = 0.0 if users_per_second is None else float(users_per_second)
    return f"Avg user inference time: {avg:.6f}s | Users/s: {ups:.2f}"


def RecallPrecision_atK(test, r, k):
    tp = r[:, :k].sum(1)
    precision = np.sum(tp) / k
    recall_n = np.array([len(test[i]) for i in range(len(test))])
    recall = np.sum(tp / recall_n)
    return precision, recall


def MRR_atK(test, r, k):
    pred = r[:, :k]
    weight = np.arange(1, k + 1)
    mrr = np.sum(pred / weight, axis=1) / np.array(
        [len(test[i]) if len(test[i]) <= k else k for i in range(len(test))]
    )
    return np.sum(mrr)


def MAP_atK(test, r, k):
    pred = r[:, :k]
    rank = pred.copy()
    for i in range(k):
        rank[:, k - i - 1] = np.sum(rank[:, : k - i], axis=1)
    weight = np.arange(1, k + 1)
    ap = np.sum(pred * rank / weight, axis=1)
    ap = ap / np.array([len(test[i]) if len(test[i]) <= k else k for i in range(len(test))])
    return np.sum(ap)


def NDCG_atK(test, r, k):
    pred = r[:, :k]
    test_mat = np.zeros((len(pred), k))
    for i, items in enumerate(test):
        length = k if k <= len(items) else len(items)
        test_mat[i, :length] = 1
    idcg = np.sum(test_mat * (1.0 / np.log2(np.arange(2, k + 2))), axis=1)
    idcg[idcg == 0.0] = 1.0
    dcg = pred * (1.0 / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.0
    return np.sum(ndcg)


def getLabel(test, pred):
    r = []
    for i in range(len(test)):
        ground_truth, pred_topk = test[i], pred[i]
        hits = np.array([item in ground_truth for item in pred_topk]).astype("float")
        r.append(hits)
    return np.array(r).astype("float")


def _init_metric_dict(topk):
    return {
        "NDCG": np.zeros(len(topk)),
        "Recall": np.zeros(len(topk)),
        "MRR": np.zeros(len(topk)),
        "MAP": np.zeros(len(topk)),
        "Precision": np.zeros(len(topk)),
    }


def _accumulate_metrics(results, ground_truth, ratings_k, topk):
    r = getLabel(ground_truth, ratings_k)
    per_user = {}
    for j, k in enumerate(topk):
        pre, rec = RecallPrecision_atK(ground_truth, r, k)
        mrr = MRR_atK(ground_truth, r, k)
        map_score = MAP_atK(ground_truth, r, k)
        ndcg = NDCG_atK(ground_truth, r, k)
        results["NDCG"][j] += ndcg
        results["Recall"][j] += rec
        results["Precision"][j] += pre
        results["MRR"][j] += mrr
        results["MAP"][j] += map_score
        per_user[f"N@{k}"] = round(ndcg, 4)
        per_user[f"R@{k}"] = round(rec, 4)
    return per_user


def _finalize_metrics(results, num_users, total_forward_time_sec=None):
    if num_users == 0:
        if total_forward_time_sec is not None:
            results["avg_user_inference_time_sec"] = 0.0
            results["users_per_second"] = 0.0
        return results
    for key in results:
        results[key] /= float(num_users)
    if total_forward_time_sec is not None:
        avg = total_forward_time_sec / float(num_users) if num_users > 0 else 0.0
        results["avg_user_inference_time_sec"] = avg
        results["users_per_second"] = 1.0 / avg if avg > 0 else 0.0
    return results


def _build_moe_stats(user_head_combinations, num_heads):
    head_usage_count = Counter()
    head_weight_sum = Counter()
    active_counts = []
    cooccurrence = np.zeros((num_heads, num_heads), dtype=np.float32)

    for active_heads in user_head_combinations:
        active_counts.append(len(active_heads))
        for head_idx, weight in active_heads:
            head_usage_count[head_idx] += 1
            head_weight_sum[head_idx] += float(weight)
        head_ids = [head_idx for head_idx, _ in active_heads]
        for i in range(len(head_ids)):
            for j in range(i + 1, len(head_ids)):
                cooccurrence[head_ids[i], head_ids[j]] += 1
                cooccurrence[head_ids[j], head_ids[i]] += 1

    total_users = len(user_head_combinations)
    all_head_ids = list(range(num_heads))
    return {
        "head_usage_count": {i: head_usage_count.get(i, 0) for i in all_head_ids},
        "head_usage_percentage": {
            i: (100.0 * head_usage_count.get(i, 0) / total_users if total_users > 0 else 0.0)
            for i in all_head_ids
        },
        "head_avg_weight": {
            i: (
                head_weight_sum.get(i, 0.0) / head_usage_count.get(i, 1)
                if head_usage_count.get(i, 0) > 0
                else 0.0
            )
            for i in all_head_ids
        },
        "avg_active_heads_per_user": float(np.mean(active_counts)) if active_counts else 0.0,
        "std_active_heads_per_user": float(np.std(active_counts)) if active_counts else 0.0,
        "min_active_heads_per_user": float(np.min(active_counts)) if active_counts else 0.0,
        "max_active_heads_per_user": float(np.max(active_counts)) if active_counts else 0.0,
        "most_common_head_combinations": Counter(
            tuple(head_idx for head_idx, _ in active_heads) for active_heads in user_head_combinations
        ).most_common(10),
        "head_cooccurrence_matrix": cooccurrence,
    }


@torch.no_grad()
def evaluate_model_improved(
    model,
    dataset,
    task_type: str = "sequential",
    eval_over_candidate_items: bool = False,
    topk: list = [1, 5, 10, 20, 50],
    return_all_user_eval_res: bool = False,
    partition: str = "test",
    sampled_user_idxes: list = None,
    batch_size: int = 32,
    include_user_history_in_eval_res: bool = False,
    return_user_moe_assignments: bool = False,
):
    del task_type, include_user_history_in_eval_res
    if partition not in {"test", "valid"}:
        raise ValueError("partition must be 'test' or 'valid'")

    model.eval()
    test_data = dataset.testData if partition == "test" else dataset.valData
    users = sampled_user_idxes if sampled_user_idxes is not None else sorted(list(test_data.keys()))
    users = [u for u in users if len(test_data[u]) > 0]

    all_item_results = _init_metric_dict(topk)
    candidate_results = _init_metric_dict(topk) if eval_over_candidate_items else None
    total_forward_time_sec = 0.0
    user_results = []
    user_moe_assignments = []
    all_head_stats = []
    num_heads = None

    for start in tqdm(range(0, len(users), batch_size), desc="Evaluating"):
        batch_users = users[start : start + batch_size]
        batch_histories = [test_data[u][0] for u in batch_users]
        batch_targets = [test_data[u][1] for u in batch_users]
        max_len = max(max(len(seq) for seq in batch_histories), 2)

        batch_inputs, batch_masks = [], []
        for seq in batch_histories:
            pad_len = max_len - len(seq)
            batch_inputs.append([0] * pad_len + seq)
            batch_masks.append([0] * pad_len + [1] * len(seq))

        batch_inputs_tensor = torch.LongTensor(batch_inputs).cuda()
        batch_masks_tensor = torch.FloatTensor(batch_masks).cuda()

        start_time = time.perf_counter()
        outputs = model.predict(batch_inputs_tensor, batch_masks_tensor)
        total_forward_time_sec += time.perf_counter() - start_time

        if isinstance(outputs, tuple):
            ratings = outputs[-1]
            batch_exit_stats = None
        elif isinstance(outputs, dict):
            ratings = outputs["all_pred_head_outputs"]
            batch_exit_stats = outputs.get("exit_head_stats")
            if batch_exit_stats is not None and num_heads is None:
                num_heads = outputs["exit_head_weight_mat"].shape[1]
        else:
            raise TypeError("Unsupported predict() output type.")

        for i, user_id in enumerate(batch_users):
            target = batch_targets[i]
            user_history = [batch_histories[i]]
            ground_truth_all = [[target]]

            all_item_ratings = ratings[i : i + 1].clone()
            exclude_index, exclude_items = [], []
            for range_i, its in enumerate(user_history):
                exclude_index.extend([range_i] * len(its))
                exclude_items.extend(its)
            all_item_ratings[exclude_index, exclude_items] = -(1 << 10)
            _, ratings_k = torch.topk(all_item_ratings, k=topk[-1])
            per_user = _accumulate_metrics(all_item_results, ground_truth_all, ratings_k.cpu().numpy(), topk)

            if eval_over_candidate_items:
                selected_items = [[target] + dataset.allPos[user_id]]
                candidate_ratings = torch.gather(
                    ratings[i : i + 1],
                    1,
                    torch.LongTensor(selected_items).cuda(),
                )
                _, candidate_ratings_k = torch.topk(candidate_ratings, k=topk[-1])
                _accumulate_metrics(candidate_results, [[0]], candidate_ratings_k.cpu().numpy(), topk)

            if return_all_user_eval_res:
                user_row = {"user_id": user_id}
                user_row.update(per_user)
                user_results.append(user_row)

            if batch_exit_stats is not None:
                all_head_stats.append(batch_exit_stats[i])
                if return_user_moe_assignments:
                    user_moe_assignments.append({"user_id": user_id, "heads": batch_exit_stats[i]})

    num_users = len(users)
    all_item_results = _finalize_metrics(all_item_results, num_users, total_forward_time_sec)
    if candidate_results is not None:
        candidate_results = _finalize_metrics(candidate_results, num_users)

    if all_head_stats:
        moe_stats = _build_moe_stats(all_head_stats, num_heads)
        all_item_results["moe_head_stats"] = moe_stats
        if candidate_results is not None:
            candidate_results["moe_head_stats"] = moe_stats

    if candidate_results is not None:
        results = {
            "all_item_ranking": all_item_results,
            "candidate_item_ranking": candidate_results,
        }
    else:
        results = all_item_results

    if return_all_user_eval_res and return_user_moe_assignments:
        return results, pd.DataFrame(user_results), user_moe_assignments
    if return_all_user_eval_res:
        return results, pd.DataFrame(user_results)
    if return_user_moe_assignments:
        return results, user_moe_assignments
    return results


@torch.no_grad()
def evaluate_multihead_model_pretrain_optimized(
    model,
    dataset,
    eval_over_candidate_items: bool = False,
    topk: list = [1, 5, 10, 20, 50],
    return_all_user_eval_res: bool = False,
    batch_size: int = 32,
):
    model.eval()
    num_intermediate_heads = len(model.intermediate_heads)
    num_total_heads = num_intermediate_heads + 1
    all_heads_results = {}
    head_forward_time_sum_sec = {}
    for head_idx in range(num_total_heads):
        head_name = f"intermediate_head_{head_idx}" if head_idx < num_intermediate_heads else "final_head"
        all_heads_results[head_name] = _init_metric_dict(topk)
        head_forward_time_sum_sec[head_name] = 0.0

    users = [u for u in range(dataset.n_user) if len(dataset.testData[u]) > 0]
    all_heads_user_results = {k: [] for k in all_heads_results} if return_all_user_eval_res else None

    for start in tqdm(range(0, len(users), batch_size), desc="Evaluating multi-head model"):
        batch_users = users[start : start + batch_size]
        batch_histories = [dataset.testData[u][0] for u in batch_users]
        batch_targets = [dataset.testData[u][1] for u in batch_users]
        max_len = max(max(len(seq) for seq in batch_histories), 2)

        batch_inputs, batch_masks = [], []
        for seq in batch_histories:
            pad_len = max_len - len(seq)
            batch_inputs.append([0] * pad_len + seq)
            batch_masks.append([0] * pad_len + [1] * len(seq))

        batch_inputs_tensor = torch.LongTensor(batch_inputs).cuda()
        batch_masks_tensor = torch.FloatTensor(batch_masks).cuda()
        outputs = model.predict(
            batch_inputs_tensor,
            batch_masks_tensor,
            return_inference_time_by_head=True,
        )
        all_pred_head_outputs = outputs["all_pred_head_outputs"]
        batch_time_by_head = outputs.get("inference_time_by_head_sec")
        if batch_time_by_head is not None:
            for head_idx in range(num_total_heads):
                head_name = f"intermediate_head_{head_idx}" if head_idx < num_intermediate_heads else "final_head"
                if head_idx < len(batch_time_by_head) and batch_time_by_head[head_idx] is not None:
                    head_forward_time_sum_sec[head_name] += float(batch_time_by_head[head_idx])

        for i, user_id in enumerate(batch_users):
            target = batch_targets[i]
            history = [batch_histories[i]]
            exclude_index, exclude_items = [], []
            for range_i, its in enumerate(history):
                exclude_index.extend([range_i] * len(its))
                exclude_items.extend(its)

            if eval_over_candidate_items:
                selected_items = [[target] + dataset.allPos[user_id]]
                ground_truth = [[0]]
            else:
                selected_items = None
                ground_truth = [[target]]

            for head_idx in range(num_total_heads):
                head_name = f"intermediate_head_{head_idx}" if head_idx < num_intermediate_heads else "final_head"
                head_ratings = all_pred_head_outputs[head_idx][i : i + 1].clone()
                head_ratings[exclude_index, exclude_items] = -(1 << 10)
                if selected_items is not None:
                    head_ratings = torch.gather(
                        head_ratings,
                        1,
                        torch.LongTensor(selected_items).cuda(),
                    )
                _, ratings_k = torch.topk(head_ratings, k=topk[-1])
                per_user = _accumulate_metrics(
                    all_heads_results[head_name],
                    ground_truth,
                    ratings_k.cpu().numpy(),
                    topk,
                )
                if return_all_user_eval_res:
                    row = {"user_id": user_id}
                    row.update(per_user)
                    all_heads_user_results[head_name].append(row)

    for head_name in all_heads_results:
        all_heads_results[head_name] = _finalize_metrics(
            all_heads_results[head_name], len(users), head_forward_time_sum_sec[head_name]
        )

    avg_time_by_head = {
        head_name: all_heads_results[head_name]["avg_user_inference_time_sec"]
        for head_name in all_heads_results
    }
    users_per_second_by_head = {
        head_name: all_heads_results[head_name]["users_per_second"] for head_name in all_heads_results
    }
    all_heads_results["avg_user_inference_time_sec_by_head"] = avg_time_by_head
    all_heads_results["users_per_second_by_head"] = users_per_second_by_head

    if return_all_user_eval_res:
        return results_with_user_dfs(all_heads_results, all_heads_user_results, topk)
    return all_heads_results


def results_with_user_dfs(all_heads_results, all_heads_user_results, topk):
    all_heads_user_dfs = {}
    columns = ["user_id"]
    for k in topk:
        columns.extend([f"N@{k}", f"R@{k}"])
    for head_name, rows in all_heads_user_results.items():
        df = pd.DataFrame(rows)
        all_heads_user_dfs[head_name] = df[columns]
    return all_heads_results, all_heads_user_dfs


def display_and_save_results(results, topk, filename=None, extra_text: str = None):
    columns = [f"@{k}" for k in topk]
    metric_results = {k: v for k, v in results.items() if _is_metric_vector(v, len(topk))}
    df = pd.DataFrame(metric_results, index=columns).T
    pd.set_option("display.float_format", "{:.4f}".format)

    title = "--- Overall Model Evaluation Results ---"
    print(f"\n{title}")
    print(df)
    inference_line = _format_inference_summary(
        results.get("avg_user_inference_time_sec"),
        results.get("users_per_second"),
    )
    if inference_line is not None:
        print(inference_line)

    if filename:
        full_output = f"{title}\n{df.to_string()}"
        if inference_line is not None:
            full_output += f"\n{inference_line}"
        if extra_text is not None:
            full_output += f"\n{extra_text}"
        with open(filename, "w") as f:
            f.write(full_output)
    return df.to_string()


def display_and_save_multihead_results_pretrain(all_heads_results, topk, filename=None):
    columns = [f"@{k}" for k in topk]
    full_output = "--- Multi-Head Model Evaluation Results ---\n\n"
    avg_time_by_head = all_heads_results.get("avg_user_inference_time_sec_by_head", {})
    ups_by_head = all_heads_results.get("users_per_second_by_head", {})

    for head_name, results in all_heads_results.items():
        if not isinstance(results, dict) or not _is_metric_vector(results.get("NDCG"), len(topk)):
            continue
        df = pd.DataFrame(results, index=columns).T
        pd.set_option("display.float_format", "{:.4f}".format)
        title = f"=== {head_name.upper()} ==="
        print(f"\n{title}")
        print(df)
        inference_line = _format_inference_summary(
            avg_time_by_head.get(head_name),
            ups_by_head.get(head_name),
        )
        if inference_line is not None:
            print(inference_line)
        full_output += f"{title}\n{df.to_string()}"
        if inference_line is not None:
            full_output += f"\n{inference_line}"
        full_output += "\n\n"

    if filename:
        with open(filename, "w") as f:
            f.write(full_output)


def save_user_eval_results(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
