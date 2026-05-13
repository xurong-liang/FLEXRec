import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LLM4RecWithMultiPredHead


class ACRouter(nn.Module):
    """
    Instance-aware dynamic soft-threshold router used by FLEXRec.
    """

    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        target_k: float = 4.0,
        tau: float = 10.0,
        gamma: float = 2.0,
        eps: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.target_k = target_k
        self.tau = tau
        self.gamma = gamma
        self.eps = eps

        self.w_gate = nn.Parameter(torch.empty(input_dim, num_experts))
        nn.init.normal_(self.w_gate, mean=0.0, std=0.006)

    def compute_two_sided_hinge_target_k_loss(self, prob_active):
        active_count_per_sample = prob_active.sum(dim=-1)
        upper_penalty = F.relu(active_count_per_sample - self.target_k)
        lower_penalty = F.relu(1.0 - active_count_per_sample)
        return torch.mean(upper_penalty + self.gamma * lower_penalty)

    def compute_continuous_load_balancing_loss(self, raw_logits):
        proxy_probs = F.softmax(raw_logits, dim=-1)
        p_i = proxy_probs.mean(dim=0)
        return self.num_experts * torch.sum(p_i * p_i)

    def forward(self, gate_inputs):
        logits = gate_inputs @ self.w_gate
        max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
        shift = torch.minimum(max_logits - self.eps, torch.zeros_like(max_logits)).detach()
        safe_logits = logits - shift

        sparse_weights = F.relu(safe_logits)
        gate_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-9)

        if self.training:
            prob_active = torch.sigmoid(self.tau * safe_logits)
            target_k_loss = self.compute_two_sided_hinge_target_k_loss(prob_active)
            lb_loss = self.compute_continuous_load_balancing_loss(logits)
            z_loss = torch.mean(logits ** 2)
        else:
            zero_tensor = torch.tensor(0.0, device=gate_inputs.device)
            target_k_loss, lb_loss, z_loss = zero_tensor, zero_tensor, zero_tensor

        return {
            "gating_weight_mat": gate_weights,
            "target_k_loss": target_k_loss,
            "lb_loss": lb_loss,
            "z_loss": z_loss,
        }


class FLEXRec(LLM4RecWithMultiPredHead):
    """
    Layer-wise fusion recommender built on LLM4RecWithMultiPredHead.
    """

    def __init__(
        self,
        target_k: float = 4.0,
        tau: float = 10.0,
        _lambda: float = 0.1,
        alpha: float = 0.1,
        pred_loss_weight: float = 1.0,
        beta: float = 1e-3,
        gamma: int = 2,
        eps: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_k = target_k
        self.tau = tau
        self._lambda = _lambda
        self.alpha = alpha
        self.beta = beta
        self.pred_loss_weight = pred_loss_weight
        self.gamma = gamma

        self.moe = ACRouter(
            num_experts=self.num_heads,
            input_dim=self.llm_model.config.hidden_size,
            target_k=self.target_k,
            tau=self.tau,
            gamma=self.gamma,
            eps=eps,
        )

    def get_llm_input_embs(self, inputs, inputs_mask):
        bs = inputs.shape[0]
        instruct_embeds = self.llm_model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
        response_embeds = self.llm_model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        item_embs = self.input_proj(self.input_embeds(inputs))
        masked_item_embs = item_embs * inputs_mask.unsqueeze(-1)
        sum_embs = masked_item_embs.sum(dim=1)
        valid_counts = inputs_mask.sum(dim=1, keepdim=True)
        raw_seq_embs = sum_embs / (valid_counts + 1e-8)

        input_embeds = torch.cat([instruct_embeds, item_embs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        return input_embeds, attention_mask, raw_seq_embs

    @staticmethod
    def _compute_exit_stats(exit_head_weight_mat):
        batch_size, _ = exit_head_weight_mat.shape
        exit_head_stats = []
        nonzero_mask = exit_head_weight_mat > 0
        for batch_idx in range(batch_size):
            active_heads = nonzero_mask[batch_idx].nonzero(as_tuple=True)[0]
            if len(active_heads) > 0:
                weights = exit_head_weight_mat[batch_idx, active_heads]
                exit_head_stats.append(list(zip(active_heads.tolist(), weights.tolist())))
            else:
                exit_head_stats.append([])
        return exit_head_stats

    def predict(self, inputs, inputs_mask) -> dict:
        batch_size = inputs.size(0)
        input_states, attention_mask, raw_seq_embs = self.get_llm_input_embs(inputs, inputs_mask)
        outputs = self.llm_model.forward_single_layer(0, input_states, attention_mask, position_ids=None)

        gate_inputs = outputs["hidden_states_with_norm"][:, -1]
        moe_out = self.moe(gate_inputs)
        exit_head_weight_mat = moe_out["gating_weight_mat"]

        active_heads_mask = (exit_head_weight_mat > 0).any(dim=0)
        active_head_indices = active_heads_mask.nonzero(as_tuple=True)[0]
        if len(active_head_indices) == 0:
            raise ValueError("No active heads found in the MoE router. Please check the router configuration.")
        else:
            max_active_head = active_head_indices[-1].item()

        all_head_outputs = torch.zeros(
            batch_size,
            self.num_heads,
            self.output_dim,
            device=inputs.device,
            dtype=torch.float32,
        )

        current_head_idx = 0
        if self.exit_interval == 1 and current_head_idx in active_head_indices:
            all_head_outputs[:, current_head_idx, :] = self.intermediate_heads[current_head_idx](
                outputs["hidden_states_with_norm"][:, -1]
            )
        if self.exit_interval == 1:
            current_head_idx += 1

        for layer_idx in range(1, self.llm_model.config.num_hidden_layers):
            if current_head_idx > max_active_head:
                break
            outputs = self.llm_model.forward_single_layer(
                layer_idx,
                outputs["hidden_states_wo_norm"],
                attention_mask=outputs["causal_mask"],
                position_ids=None,
            )
            if (layer_idx + 1) % self.exit_interval == 0 and current_head_idx < self.num_heads:
                if current_head_idx in active_head_indices:
                    if current_head_idx < self.num_heads - 1:
                        head_output = self.intermediate_heads[current_head_idx](
                            outputs["hidden_states_with_norm"][:, -1]
                        )
                    else:
                        head_output = self.score(outputs["hidden_states_with_norm"][:, -1])
                    all_head_outputs[:, current_head_idx, :] = head_output
                current_head_idx += 1

        weighted_outputs = all_head_outputs * exit_head_weight_mat.unsqueeze(-1)
        final_exit_outputs = weighted_outputs.sum(dim=1)

        return {
            "all_pred_head_outputs": final_exit_outputs,
            "exit_head_weight_mat": exit_head_weight_mat,
            "moe_output": moe_out,
            "exit_head_stats": self._compute_exit_stats(exit_head_weight_mat),
            "gate_inputs": gate_inputs,
            "raw_seq_embs": raw_seq_embs,
        }

    def forward(self, inputs, inputs_mask, user_history, labels) -> dict:
        del user_history
        if not self.training:
            raise ValueError("FLEXRec.forward() should only be used during training.")

        outputs = self.predict(inputs, inputs_mask)
        moe_out = outputs["moe_output"]
        pred_loss = nn.CrossEntropyLoss()(outputs["all_pred_head_outputs"], labels.view(-1))
        pred_loss = pred_loss * self.pred_loss_weight
        total_loss = (
            pred_loss
            + (self._lambda * moe_out["target_k_loss"])
            + (self.alpha * moe_out["lb_loss"])
            + (self.beta * moe_out["z_loss"])
        )
        return {
            **outputs,
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "target_k_loss": moe_out["target_k_loss"],
            "lb_loss": moe_out["lb_loss"],
            "z_loss": moe_out["z_loss"],
        }
