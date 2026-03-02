import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, List
from src.core.signal import InternalSignal
from src.processor.base import Processor


class MatchOutputProcessor(Processor):
    def __init__(self, core_dim=128):
        super().__init__()
        self.decoder = nn.Linear(core_dim, 2)
        self.optimizer = optim.Adam(self.decoder.parameters(), lr=0.005)
        self.loss_fn = nn.CrossEntropyLoss()
        self.saved_logits = None

    def process(self, intent_signals: List[InternalSignal]) -> None:
        if not intent_signals:
            return

        concept = intent_signals[0].vector

        logits = self.decoder(concept)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        self.saved_logits = logits

        idx = torch.argmax(logits).item()
        res = "MATCH" if idx == 0 else "MISMATCH"
        self.output_buffer.append(res)

    def learn(self, is_match: bool):
        if self.saved_logits is None:
            return None
        target = torch.tensor([0 if is_match else 1], dtype=torch.long)
        loss = self.loss_fn(self.saved_logits, target)
        self.saved_logits = None
        return loss

    def parameters(self) -> List[Any]:
        return super().parameters()
