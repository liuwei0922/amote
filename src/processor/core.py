# src/processor/core.py
import torch
import torch.nn as nn
from typing import List
from src.core.signal import InternalSignal
from src.processor.base import Processor


class CoreProcessor(Processor):
    def __init__(self, core_dim=128):
        super().__init__()
        self.dim = core_dim

        self.fusion_net = nn.Sequential(
            nn.Linear(core_dim * 2, core_dim * 2),
            nn.ReLU(),
            nn.Linear(core_dim * 2, core_dim),
        )

    def process(self, signals: List[InternalSignal]) -> None:
        if len(signals) < 2:
            return

        t_tensor = signals[0].vector
        s_tensor = signals[1].vector

        cat_vec = torch.cat([t_tensor, s_tensor], dim=0)
        fused_vec = self.fusion_net(cat_vec)

        fused_vec = torch.nn.functional.normalize(fused_vec, p=2, dim=0)

        sig = InternalSignal(vector=fused_vec)

        self.output_buffer.append(sig)

    def parameters(self) -> List[torch.nn.Parameter]:
        return list(self.fusion_net.parameters())