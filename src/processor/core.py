import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, List
from src.core.signal import InternalSignal
from src.processor.base import Processor


class ConceptMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())

    def forward(self, state_seq):
        attn_out, _ = self.attn(state_seq, state_seq, state_seq)
        state_seq = self.norm(state_seq + attn_out)
        return self.fc(state_seq)


# ---------------------------------------------------------
# 子模块 2: 路由裁判 (Judge) - 负责分发
# ---------------------------------------------------------
class Judge(nn.Module):
    def __init__(self, dim, num_routes):
        super().__init__()
        # 简单全连接：根据当前思考状态，决定扔给哪个 OutputProcessor
        self.router_net = nn.Linear(dim, num_routes)

    def forward(self, state_seq):
        # 把思考序列池化为一个向量
        pooled = torch.mean(state_seq, dim=1)
        return self.router_net(pooled)


# ---------------------------------------------------------
# 动力学核心 (Dynamical Core)
# ---------------------------------------------------------
class DynamicalCoreProcessor(Processor):
    def __init__(self, core_dim=128):
        super().__init__()
        self.dim = core_dim

        self.mixer = ConceptMixer(core_dim)
        self.judge = None
        self.optimizer = None
        self.loss_fn = nn.CrossEntropyLoss()

        self.routes: List[Processor] = []

        self.max_think_steps = 3
        self.convergence_threshold = 0.01
        self.saved_route_logits = None

    def register_output(self, processor: Processor):
        self.routes.append(processor)
        self.judge = Judge(self.dim, len(self.routes))
        self.optimizer = optim.Adam(
            list(self.mixer.parameters()) + list(self.judge.parameters()), lr=0.002
        )

    def process(self, signals: List[InternalSignal]) -> None:
        if len(signals) < 2:
            return
        tensors = [s.vector for s in signals]
        current_state = torch.stack(tensors).unsqueeze(0)

        for _ in range(self.max_think_steps):
            prev_state = current_state.clone()
            delta = self.mixer(current_state)
            current_state = current_state + delta

            diff = torch.norm(current_state - prev_state)
            if diff.item() < self.convergence_threshold:
                break

        if self.judge and self.routes is not None:
            route_logits = self.judge(current_state)
            self.saved_route_logits = route_logits
            route_idx = torch.argmax(route_logits).item()

            final_concept = torch.mean(current_state, dim=1).squeeze(0)

            sig = InternalSignal(vector=final_concept, debug_info="Intent")

            target_processor = self.routes[route_idx]
            target_processor.process([sig])

    def learn(self, target_route_idx: int):
        if self.saved_route_logits is None:
            return None
        target = torch.tensor([target_route_idx], dtype=torch.long)
        loss = self.loss_fn(self.saved_route_logits, target)
        self.saved_route_logits = None
        return loss

    def parameters(self) -> List[Any]:
        return super().parameters()
