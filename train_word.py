import torch
import torch.nn as nn
import torch.optim as optim
import pickle


def build_torch_vocab():
    print("=== 开始用 PyTorch 纯手工炼制单字向量 ===")

    corpus_text = """
    小明昨天在超市买了一个红苹果
    超市里有很多水果苹果很甜
    昨天天气很好小明去买水果
    红色的苹果很好吃
    小明喜欢吃红苹果
    在超市可以买到很多水果
    """

    chars = [c for c in corpus_text if c.strip() and c not in "。，！？"]

    unique_chars = list(set(chars))
    char_to_ix = {ch: i for i, ch in enumerate(unique_chars)}
    ix_to_char = {i: ch for i, ch in enumerate(unique_chars)}
    vocab_size = len(unique_chars)

    print(f"语料库词汇量: {vocab_size} 个不重复单字")

    WINDOW_SIZE = 2
    data = []
    for i in range(WINDOW_SIZE, len(chars) - WINDOW_SIZE):
        target = char_to_ix[chars[i]]
        context = [
            char_to_ix[chars[i - 2]],
            char_to_ix[chars[i - 1]],
            char_to_ix[chars[i + 1]],
            char_to_ix[chars[i + 2]],
        ]
        for ctx in context:
            data.append((target, ctx))

    print(f"生成了 {len(data)} 对 (中心字 -> 上下文) 训练样本")

    EMBEDDING_DIM = 128

    class SkipGramModel(nn.Module):
        def __init__(self, vocab_size, emb_dim):
            super(SkipGramModel, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, emb_dim)
            self.linear = nn.Linear(emb_dim, vocab_size)

        def forward(self, target_idx):
            embeds = self.embeddings(target_idx)
            out = self.linear(embeds)
            return out

    model = SkipGramModel(vocab_size, EMBEDDING_DIM)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 200
    print("\n开始反向传播训练...")
    for epoch in range(epochs):
        total_loss = 0
        for target, context in data:
            target_tensor = torch.tensor([target], dtype=torch.long)
            context_tensor = torch.tensor([context], dtype=torch.long)

            model.zero_grad()
            log_probs = model(target_tensor)
            loss = loss_function(log_probs, context_tensor)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} - 损失 Loss: {total_loss/len(data):.4f}"
            )

    vocab_dict = {}
    embeddings_matrix = model.embeddings.weight.data.numpy()

    for char, idx in char_to_ix.items():

        vec = embeddings_matrix[idx]
        norm = torch.linalg.norm(torch.tensor(vec)).item()
        vocab_dict[char] = vec / (norm + 1e-8)

    vocab_path = "pretrained_chars_128d.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab_dict, f)

    print(f"\n✅ 成功纯手工炼制并保存 128 维向量 -> {vocab_path}")

    def cosine_sim(v1, v2):
        return torch.nn.functional.cosine_similarity(
            torch.tensor(v1), torch.tensor(v2), dim=0
        ).item()

    try:
        sim_apple = cosine_sim(vocab_dict["苹"], vocab_dict["果"])
        print(f"['苹' 和 '果'] 的余弦相似度: {sim_apple:.3f}")

        sim_random = cosine_sim(vocab_dict["苹"], vocab_dict["昨"])
        print(f"['苹' 和 '昨'] 的余弦相似度: {sim_random:.3f}")
    except KeyError:
        pass


if __name__ == "__main__":
    build_torch_vocab()
