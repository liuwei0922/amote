import torch
import torch.nn as nn
from .processor.input.vec import TextInputProcessor
from .processor.input.state import StateInputProcessor
from .processor.output.match import MatchOutputProcessor
from .processor.core import CoreProcessor


class Router(nn.Module):
    def __init__(self, core_dim, seq_len, num_outputs):
        super().__init__()
        input_size = seq_len * core_dim

        self.selector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs),
        )

        self.arg_generator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, core_dim),
            nn.LayerNorm(core_dim),
        )

    def forward(self, internal_thoughts):
        selection_logits = self.selector(internal_thoughts)  # [B, N]
        output_arg = self.arg_generator(internal_thoughts)  # [B, D]

        return selection_logits, output_arg


class System(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 128

        self.inputs = nn.ModuleList(
            [
                TextInputProcessor(core_dim=self.dim),
                StateInputProcessor(core_dim=self.dim),
            ]
        )

        self.outputs = nn.ModuleList([MatchOutputProcessor(core_dim=self.dim)])

        self.core = CoreProcessor(core_dim=self.dim)

        self.seq_len = 2
        self.router = Router(self.dim, self.seq_len, len(self.outputs))

    def forward(self, input_data_list):
        tensor_list = []
        for i, processor in enumerate(self.inputs):
            data = input_data_list[i]
            out = processor(data)  # [B, 1, D]
            tensor_list.append(out)

        mixed_input = torch.cat(tensor_list, dim=1)  # [B, 2, D]

        internal_thoughts = self.core(mixed_input)  # [B, 2, D]

        route_logits, output_arg = self.router(internal_thoughts)

        if self.training:
            results = []
            for output_proc in self.outputs:
                res = output_proc(output_arg)
                results.append(res)

            return results, route_logits

        else:
            chosen_idx = int(torch.argmax(route_logits, dim=1).item())

            target_proc = self.outputs[chosen_idx]
            res = target_proc(output_arg)

            results = [None] * len(self.outputs)
            results[chosen_idx] = res

            return results, route_logits

    def consolidate_memory(self, correct_mask):
        self.core.update_memory(correct_mask)
