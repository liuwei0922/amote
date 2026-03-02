import numpy as np
import torch
from typing import List
from src.core.signal import InternalSignal
from src.processor.base import Processor

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]


class ActionOutputProcessor(Processor):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def process(self, intent_signals: List[InternalSignal]) -> None:
        for sig in intent_signals:
            vec = sig.vector

            if vec is None:
                continue
            action_idx = np.argmax(vec)
            physical_action = ACTIONS[action_idx]

            self.output_buffer.append(physical_action)
            # print(f"  [Output] 解码意图信号 -> 物理动作: {physical_action}")
