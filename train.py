import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import random


DIM = 128
try:
    with open("pretrained_chars_128d.pkl", "rb") as f:
        vocab_dict = pickle.load(f)
    print(f"成功加载单字向量字典: {len(vocab_dict)} 字")
except FileNotFoundError:
    print("错误：没找到 pretrained_chars_128d.pkl，请先运行 build_vocab_torch.py")
    exit()


def text_to_tensor(text):
    """把字符串变成 [1, Len, DIM] 的 Tensor"""
    vectors = []
    for char in text:
        if char in vocab_dict:
            vectors.append(torch.tensor(vocab_dict[char], dtype=torch.float32))
        else:
            vectors.append(torch.zeros(DIM))
    if not vectors:
        return None
    return torch.stack(vectors).unsqueeze(0)


class ConceptAutoEncoder(nn.Module):
    def __init__(self, dim, max_stack=8):
        super().__init__()
        self.dim = dim
        self.max_stack = max_stack

        self.encoder_fc = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim)
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, max_stack * dim),
        )

    def forward(self, x_seq):
        """
        x_seq: [1, seq_len, DIM]
        注意：为了简单，我们训练时假设输入已经 Padding 到 max_stack，或者我们只取平均
        """

        mean_vec = torch.mean(x_seq, dim=1)  # [1, DIM]
        concept_vector = self.encoder_fc(mean_vec)
        concept_vector = torch.nn.functional.normalize(concept_vector, p=2, dim=1)

        reconstructed_flat = self.decoder_fc(concept_vector)

        reconstructed = reconstructed_flat.view(1, self.max_stack, self.dim)

        return concept_vector, reconstructed


def train():
    MAX_STACK = 8
    model = ConceptAutoEncoder(dim=DIM, max_stack=MAX_STACK)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    mse_loss = nn.MSELoss()

    corpus = [
        "小明",
        "昨天",
        "超市",
        "苹果",
        "水果",
        "红苹果",
        "买水果",
        "去超市",
        "天气",
        "小明昨天",
        "在超市买",
    ]

    print("\n>>> 开始无监督自编码训练 (Auto-Encoding)...")
    epochs = 500

    for epoch in range(epochs):
        total_loss = 0.0

        for phrase in corpus:
            raw_tensor = text_to_tensor(phrase)
            if raw_tensor is None:
                continue

            curr_len = raw_tensor.size(1)
            if curr_len < MAX_STACK:
                pad = torch.zeros(1, MAX_STACK - curr_len, DIM)
                input_padded = torch.cat([raw_tensor, pad], dim=1)
            else:
                input_padded = raw_tensor[:, :MAX_STACK, :]

            optimizer.zero_grad()
            _, recon = model(input_padded)

            loss = mse_loss(recon, input_padded)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            print(f"  [Epoch {epoch+1}] 重构误差 Loss: {total_loss:.6f}")

    torch.save(model.state_dict(), "trained_brain_full.pth")
    print("\n✅ 训练完成！Encoder 权重已保存至 core_encoder_weights.pth")

    print("\n>>> 验证：对比'真词'和'乱码'的重构难度")
    model.eval()

    def get_score(phrase):

        with torch.no_grad():
            raw = text_to_tensor(phrase)
            curr_len = raw.size(1)
            pad = torch.zeros(1, MAX_STACK - curr_len, DIM)
            inp = torch.cat([raw, pad], dim=1)

            _, recon = model(inp)
            error = mse_loss(recon, inp).item()
            return 1.0 / (1.0 + error * 100) 

    test_cases = ["红苹果", "在超", "小明", "明昨", "超市", "果很"]
    print(f"{'词组':<10} | {'内聚得分':<10}")
    print("-" * 25)
    for t in test_cases:
        score = get_score(t)
        print(f"{t:<10} | {score:.4f}")


if __name__ == "__main__":
    train()
