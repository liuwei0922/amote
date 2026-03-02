import random
from src.processor.input.vec import TextInputProcessor
from src.processor.input.state import StateInputProcessor
from src.processor.core import DynamicalCoreProcessor
from src.processor.output.match import MatchOutputProcessor
import torch

WORLD_RULES = {"向左": "WEST", "向右": "EAST", "向前": "NORTH", "向后": "SOUTH"}


def main():
    print("=== AGI Pipeline: 动力学核心 + 路由分发测试 ===")

    CORE_DIM = 128

    p_text = TextInputProcessor(core_dim=CORE_DIM)
    p_state = StateInputProcessor(core_dim=CORE_DIM)

    p_core = DynamicalCoreProcessor(core_dim=CORE_DIM)

    p_out = MatchOutputProcessor(core_dim=CORE_DIM)

    p_core.register_output(p_out)  # 路由 ID 为 0

    trainable_processors = [p_text, p_state]

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

        loss_out = None
        if p_out.output_buffer:
            guess_res = p_out.fetch_signals()[0]
            loss_out = p_out.learn(is_correct)
        else:
            guess_res = "NO_ROUTE"
            
        loss_route = p_core.learn(target_route_idx=0)
        
        total_loss = torch.tensor(0.0, requires_grad=True)
        if loss_out is not None: total_loss = total_loss + loss_out
        if loss_route is not None: total_loss = total_loss + loss_route
        
        p_text.optimizer.zero_grad()
        p_state.optimizer.zero_grad()
        p_core.optimizer.zero_grad()
        p_out.optimizer.zero_grad()
        
        total_loss.backward()
        
        p_text.optimizer.step()
        p_state.optimizer.step()
        p_core.optimizer.step()
        p_out.optimizer.step()


        if (epoch + 1) % 400 == 0:
            print(
                f"Epoch {epoch+1:4d} | 听:[{inst}] 摸:[{state}] -> 猜:{guess_res} | 真:{'MATCH' if is_correct else 'MISMATCH'}"
            )

    # 测试泛化
    print("\n=== 测试泛化涌现 ===")
    test_cases = [
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
        if p_out.output_buffer:
            ans = p_out.fetch_signals()[0]
        else:
            ans = "NO_ROUTE"
        print(f"指令:'{t}', 状态:'{s}' => AI判断: {ans}")


if __name__ == "__main__":
    main()
