import torch
import torch.nn as nn
from typing import Any, List
from src.core.signal import InternalSignal
from src.processor.base import Processor


class MatchOutputProcessor(Processor):
    def __init__(self, core_dim=128):
        super().__init__()
        self.decoder = nn.Linear(core_dim, 2)

    def process(self, intent_signals: List[InternalSignal]) -> None:
        if not intent_signals:
            return

        fused_tensor = intent_signals[0].vector

        logits = self.decoder(fused_tensor)
        self.saved_logits = logits.unsqueeze(0)

        action_idx = torch.argmax(logits, dim=-1).item()
        debug_res = "MATCH" if action_idx == 0 else "MISMATCH"

        self.output_buffer.append(debug_res)

    def parameters(self) -> list:
        return list(self.decoder.parameters())
