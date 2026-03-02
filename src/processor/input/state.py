import torch
import torch.nn as nn
import numpy as np
from src.core.signal import InternalSignal
from src.processor.base import Processor


class StateInputProcessor(Processor):
    def __init__(self, core_dim=128):
        super().__init__()
        self.state_map = {"NORTH": 0, "SOUTH": 1, "EAST": 2, "WEST": 3}

        self.translator = nn.Sequential(nn.Linear(4, core_dim), nn.ReLU())
        self.optimizer = torch.optim.Adam(self.translator.parameters(), lr=0.005)

    def process(self, state: str) -> None:
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[self.state_map[state]] = 1.0
        raw_tensor = torch.tensor(one_hot)

        concept_tensor = self.translator(raw_tensor)

        signal = InternalSignal(vector=concept_tensor, debug_info=f"State:{state}")
        self.output_buffer.append(signal)
    
    def parameters(self) -> list:
        return list(self.translator.parameters())
