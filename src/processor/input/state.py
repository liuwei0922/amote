import torch
import torch.nn as nn
import numpy as np
from typing import List


class StateInputProcessor(nn.Module):
    def __init__(self, core_dim=128):
        super().__init__()
        self.state_map = {"NORTH": 0, "SOUTH": 1, "EAST": 2, "WEST": 3}
        self.translator = nn.Sequential(nn.Linear(4, core_dim), nn.ReLU())

    def forward(self, states: List[str]):
        batch_size = len(states)
        one_hots = []

        for s in states:
            one_hot = np.zeros(4, dtype=np.float32)
            if s in self.state_map:
                one_hot[self.state_map[s]] = 1.0
            one_hots.append(one_hot)

        raw_tensor = torch.tensor(np.array(one_hots))

        concept_tensor = self.translator(raw_tensor)

        return concept_tensor.unsqueeze(1)
