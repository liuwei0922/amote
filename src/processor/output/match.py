import torch
import torch.nn as nn


class MatchOutputProcessor(nn.Module):
    def __init__(self, core_dim=128):
        super().__init__()
        self.decoder = nn.Linear(core_dim, 2)

    def forward(self, concept: torch.Tensor) -> torch.Tensor:
        return self.decoder(concept)

