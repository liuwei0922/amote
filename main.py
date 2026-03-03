import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.system import System


WORLD_RULES = {"左": "WEST", "右": "EAST", "前": "NORTH", "后": "SOUTH"}
ALL_INSTRUCTIONS = list(WORLD_RULES.keys())
ALL_STATES = ["NORTH", "SOUTH", "EAST", "WEST"]


def main():
    print("=== AGI System: 端到端动态算子网络训练 ===")

    model = System()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 6000
    for epoch in range(epochs):
        if random.random() < 0.5:
            inst = random.choice(ALL_INSTRUCTIONS)
            state = WORLD_RULES[inst]
            is_match = True
        else:
            inst = random.choice(ALL_INSTRUCTIONS)
            state = random.choice(ALL_STATES)
            while state == WORLD_RULES[inst]:
                state = random.choice(ALL_STATES)
            is_match = False

        inputs = [[inst], [state]]

        optimizer.zero_grad()

        results, route_logits = model(inputs)

        logits = results[0]

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        target_label = 0 if is_match else 1
        target = torch.tensor([target_label], dtype=torch.long)

        loss = loss_fn(logits, target)

        loss.backward()
        optimizer.step()


        if (epoch + 1) % 200 == 0:
            pred_idx = torch.argmax(logits, dim=-1).item()
            
            pred_str = "MATCH" if pred_idx == 0 else "MISMATCH"
            truth_str = "MATCH" if is_match else "MISMATCH"
            
            print(f"Epoch {epoch+1:4d} | 听:[{inst}] 摸:[{state}] -> 猜:{pred_str} | 真:{truth_str} | Loss:{loss.item():.4f}")

    # --- 泛化测试 ---
    print("\n=== 泛化测试 (Zero-Shot) ===")
    test_cases = [
        ("左", "WEST", True),
        ("右", "NORTH", False),
        ("前", "NORTH", True),
        ("后", "SOUTH", True),
        ("向左", "WEST", True),
        ("向右", "EAST", True),
    ]

    model.eval()  # 切换到评估模式
    with torch.no_grad():
        for t, s, truth in test_cases:
            inputs = [[t], [s]]
            results, route_logits = model(inputs)
            logits = results[0] # Tensor
            pred_idx = torch.argmax(logits, dim=-1).item()
            ans = "MATCH" if pred_idx == 0 else "MISMATCH"
            print(
                f"指令:'{t}', 状态:'{s}' => AI判断: {ans} (预期: {'MATCH' if truth else 'MISMATCH'})"
            )


if __name__ == "__main__":
    main()
