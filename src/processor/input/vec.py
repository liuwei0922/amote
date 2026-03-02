from typing import List

import torch
import torch.nn as nn
import numpy as np
from text2vec import Word2Vec
from src.core.signal import InternalSignal
from src.processor.base import Processor

class TextInputProcessor(Processor):
    def __init__(self, core_dim=128):
        super().__init__()
        self.w2v = Word2Vec("w2v-light-tencent-chinese")
        
        self.translator = nn.Sequential(
            nn.Linear(200, core_dim),
            nn.ReLU()
        )
        self.optimizer = torch.optim.Adam(self.translator.parameters(), lr=0.005)

    def process(self, text: str) -> None:
        raw_emb = self.w2v.encode([text], show_progress_bar=False)[0]
        raw_tensor = torch.tensor(raw_emb, dtype=torch.float32)
        
        concept_tensor = self.translator(raw_tensor)
        
        signal = InternalSignal(vector=concept_tensor, debug_info=f"Text:{text}")
        self.output_buffer.append(signal)

    def parameters(self) -> List[torch.nn.Parameter]:
        return list(self.translator.parameters())