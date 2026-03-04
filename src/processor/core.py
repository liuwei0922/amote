import torch
import torch.nn as nn
from .memory import GraphMemory


class CoreProcessor(nn.Module):
    def __init__(self, core_dim=128):
        super().__init__()
        self.dim = core_dim

        self.memory = GraphMemory(dim=core_dim)

        self.fusion_net = nn.Sequential(
            nn.Linear(core_dim, core_dim), nn.LayerNorm(core_dim), nn.ReLU()  # 激活
        )

        self.op_net = nn.Linear(core_dim, core_dim)

    def forward(self, input_tensor):
        batch_size, seq_len, dim = input_tensor.size()

        features = self.fusion_net(input_tensor)
        raw_output = self.op_net(features)  # [B, S, D]

        corrected_output_list = []

        for b in range(batch_size):
            candidates = {}

            for s in range(seq_len):
                token = input_tensor[b, s, :]
                idxs, tensors, weights = self.memory.query_with_indices(
                    token, threshold=0.1
                )

                for idx, tens, w in zip(idxs, tensors, weights):
                    if idx not in candidates:
                        candidates[idx] = {"tensor": tens, "weights": [0.0] * seq_len}
                    candidates[idx]["weights"][s] = w
            final_seq = []
            for s_out in range(seq_len):
                target_vec = raw_output[b, s_out, :]

                correction = torch.zeros_like(target_vec)
                total_influence = 0.0

                for idx, data in candidates.items():
                    w_list = data["weights"]
                    mem_vec = data["tensor"].to(target_vec.device)
                    compound_weight = sum(w_list)

                    if compound_weight > 0.01:
                        norm_sq = torch.dot(mem_vec, mem_vec)
                        if norm_sq > 1e-6:
                            proj = (torch.dot(target_vec, mem_vec) / norm_sq) * mem_vec
                            correction += proj * compound_weight
                            total_influence += compound_weight

                if total_influence > 0.01:
                    correction = correction / (total_influence + 1e-5)
                    final_vec = target_vec + 0.5 * correction
                else:
                    final_vec = target_vec

                final_seq.append(final_vec)

            corrected_output_list.append(torch.stack(final_seq))

        final_output = torch.stack(corrected_output_list)
        self.last_io_pair = (input_tensor.detach(), final_output.detach())
        return final_output

    def update_memory(self, mask=None):
        if hasattr(self, "last_io_pair"):
            inputs, outputs = self.last_io_pair

            batch_size = inputs.size(0)
            seq_len = inputs.size(1)

            if mask is None:
                mask = torch.ones(batch_size, dtype=torch.bool, device=inputs.device)

            for b in range(batch_size):
                if mask[b]:
                    for s in range(seq_len):
                        if outputs.dim() == 3:
                            inp = inputs[b, s, :]
                            out = outputs[b, s, :]
                            self.memory.link(inp, out)
                        elif outputs.dim() == 2:
                            inp = inputs[b, s, :]
                            out = outputs[b, :]
                            self.memory.link(inp, out)
