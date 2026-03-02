import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.processor.input.vec import TextInputProcessor
from src.processor.input.state import StateInputProcessor
from src.processor.core import CoreProcessor
from src.processor.output.match import MatchOutputProcessor

WORLD_RULES = {"左": "WEST", "右": "EAST", "前": "NORTH", "后": "SOUTH"}


def main():
    print("=== 通用多模态 AGI 架构 (完全解耦版) ===")
    CORE_DIM = 128

    p_text = TextInputProcessor(core_dim=CORE_DIM)
    p_state = StateInputProcessor(core_dim=CORE_DIM)
    p_core = CoreProcessor(core_dim=CORE_DIM)
    p_out = MatchOutputProcessor(core_dim=CORE_DIM)

    all_params = (
        p_text.parameters()
        + p_state.parameters()
        + p_core.parameters()
        + p_out.parameters()
    )

    global_optimizer = optim.Adam(all_params, lr=0.005)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 6000
    for epoch in range(epochs):
        p_text.output_buffer.clear()
        p_state.output_buffer.clear()
        p_core.output_buffer.clear()
        p_out.output_buffer.clear()

        if random.random() < 0.5:
            inst = random.choice(list(WORLD_RULES.keys()))
            state = WORLD_RULES[inst]
            is_correct = True
        else:
            inst = random.choice(list(WORLD_RULES.keys()))
            wrong_states = ["NORTH", "SOUTH", "EAST", "WEST"]
            wrong_states.remove(WORLD_RULES[inst])
            state = random.choice(wrong_states)
            is_correct = False

        p_text.process(inst)
        p_state.process(state)

        signals_in = p_text.fetch_signals() + p_state.fetch_signals()
        p_core.process(signals_in)

        intent_signals = p_core.fetch_signals()
        p_out.process(intent_signals)

        guess_res = p_out.fetch_signals()[0]

        target = torch.tensor([0 if is_correct else 1], dtype=torch.long)
        loss = loss_fn(p_out.saved_logits, target)

        global_optimizer.zero_grad()
        loss.backward()
        global_optimizer.step()

        if (epoch + 1) % 400 == 0:
            print(
                f"Epoch {epoch+1:4d} | 听:[{inst}] 摸:[{state}] -> 猜:{guess_res} | 真:{'MATCH' if is_correct else 'MISMATCH'}"
            )

    test_cases = [
        ("向前", "NORTH"),
        ("向后", "NORTH"),
        ("向右", "EAST"),
        ("向左", "WEST"),
        ("左边", "WEST"),
        ("右", "NORTH"),
        ("前进", "NORTH"),
        ("后退", "SOUTH"),
        ("向左", "WEST"),
        ("向右", "EAST"),
    ]

    for t, s in test_cases:
        p_text.process(t)
        p_state.process(s)
        p_core.process(p_text.fetch_signals() + p_state.fetch_signals())
        intent_signals = p_core.fetch_signals()
        p_out.process(intent_signals)

        ans = p_out.fetch_signals()[0]
        print(f"指令:'{t}', 状态:'{s}' => AI判断: {ans}")


if __name__ == "__main__":
    main()
