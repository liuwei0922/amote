from typing import List
import torch
import torch.nn as nn
import numpy as np
from text2vec import Word2Vec


class TextInputProcessor(nn.Module):
    def __init__(self, core_dim=128):
        super().__init__()
        self.w2v = Word2Vec("w2v-light-tencent-chinese")

        self.translator = nn.Sequential(nn.Linear(200, core_dim), nn.ReLU())

    def forward(self, text: List[str]) -> torch.Tensor:
        raw_emb = self.w2v.encode(text, show_progress_bar=False)
        raw_tensor = torch.tensor(raw_emb, dtype=torch.float32)

        concept_tensor = self.translator(raw_tensor)
        return concept_tensor.unsqueeze(1)
