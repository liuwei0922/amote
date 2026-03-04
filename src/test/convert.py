from gensim.models import KeyedVectors
from safetensors.torch import save_file
import torch
import json

print("正在加载 Word2Vec bin...")
wv = KeyedVectors.load_word2vec_format("light_Tencent_AILab_ChineseEmbedding.bin", binary=True)

vocab = {word: i for i, word in enumerate(wv.index_to_key)}
vectors = torch.from_numpy(wv.vectors)

print("正在保存 safetensors...")
save_file({"weight": vectors}, "tencent_w2v.safetensors")

print("正在保存 vocab.json...")
with open("tencent_vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

print("搞定！现在你可以卸载 Python 了。")