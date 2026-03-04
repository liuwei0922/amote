import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from src.system import System

WORLD_RULES = {"左": "WEST", "右": "EAST", "前": "NORTH", "后": "SOUTH"}
ALL_INSTRUCTIONS = list(WORLD_RULES.keys())
ALL_STATES = ["NORTH", "SOUTH", "EAST", "WEST"]


BATCH_SIZE = 32
EPOCHS = 5000


def main():
    print("=== AGI System: 端到端动态算子网络训练 (带图记忆 + 可视化) ===")

    model = System()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.CrossEntropyLoss()

    history_loss = []
    history_acc = []

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        batch_insts = []
        batch_states = []
        batch_targets = []

        for _ in range(BATCH_SIZE):
            if random.random() < 0.5:
                inst = random.choice(ALL_INSTRUCTIONS)
                state = WORLD_RULES[inst]
                label = 0
            else:
                inst = random.choice(ALL_INSTRUCTIONS)
                state = random.choice(ALL_STATES)
                while state == WORLD_RULES[inst]:
                    state = random.choice(ALL_STATES)
                label = 1

            batch_insts.append(inst)
            batch_states.append(state)
            batch_targets.append(label)

        inputs = [batch_insts, batch_states]
        targets = torch.tensor(batch_targets, dtype=torch.long)

        results, _ = model(inputs)
        logits = results[0]

        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        correct_mask = preds == targets
        model.consolidate_memory(correct_mask)
        acc = (preds == targets).float().mean().item()

        history_loss.append(loss.item())
        history_acc.append(acc)

        if (epoch + 1) % 200 == 0:
            print(
                f"Epoch {epoch+1:4d} | Batch Loss:{loss.item():.4f} | Batch Acc:{acc:.2f}"
            )

    # --- 画图 ---
    print("\n>>> 正在生成训练曲线图 (training_curve.png)...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_loss, alpha=0.3, color="blue", label="Raw Loss")
    # 平滑 Loss 曲线
    smooth_loss = [
        sum(history_loss[max(0, i - 50) : i + 1])
        / len(history_loss[max(0, i - 50) : i + 1])
        for i in range(len(history_loss))
    ]
    plt.plot(smooth_loss, color="red", label="Smooth Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_acc, color="green", label="Rolling Accuracy (Window=100)")
    plt.axhline(y=0.5, color="gray", linestyle="--")  # 50% 瞎猜线
    plt.axhline(y=1.0, color="gray", linestyle="--")  # 100% 完美线
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("training_curve.png")
    print("✅ 图片已保存！请查看 training_curve.png")

    # --- 泛化测试 (保持不变) ---
    print("\n=== 泛化测试 (Zero-Shot) ===")
    test_cases = [
        ("左", "WEST", True),
        ("右", "NORTH", False),
        ("前", "NORTH", True),
        ("后", "SOUTH", True),
        ("向左", "WEST", True),
        ("向右", "EAST", True),
    ]
    model.eval()
    with torch.no_grad():
        for t, s, truth in test_cases:
            inputs = [[t], [s]]
            results, route_logits = model(inputs)
            logits = results[0]
            pred_idx = torch.argmax(logits, dim=-1).item()
            ans = "MATCH" if pred_idx == 0 else "MISMATCH"
            print(
                f"指令:'{t}', 状态:'{s}' => AI判断: {ans} (预期: {'MATCH' if truth else 'MISMATCH'})"
            )


if __name__ == "__main__":
    main()
