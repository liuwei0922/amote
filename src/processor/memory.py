import torch
import numpy as np


class GraphMemory:
    def __init__(self, dim=128):
        self.nodes = []

        self.adj = {}

        self.dim = dim

    def _get_node_index(self, tensor):
        if not self.nodes:
            return -1

        stack = torch.stack(self.nodes)
        dists = torch.norm(stack - tensor, dim=1)
        min_dist, idx = torch.min(dists, dim=0)

        if min_dist < 0.1:
            return idx.item()
        return -1

    def register(self, tensor):
        idx = self._get_node_index(tensor)
        if idx != -1:
            return idx

        self.nodes.append(tensor.detach().clone())
        idx = len(self.nodes) - 1
        self.adj[idx] = {}
        return idx

    def link(self, input_tensor, output_tensor, weight=1.0):
        src = self.register(input_tensor)
        dst = self.register(output_tensor)

        current_w = self.adj[src].get(dst, 0.0)
        self.adj[src][dst] = current_w + weight

    def query(self, input_tensor, top_k=5):
        src = self._get_node_index(input_tensor)
        if src == -1:
            return [], []

        candidates = []
        if src in self.adj:
            for dst, w in self.adj[src].items():
                candidates.append((w, self.nodes[dst]))

        candidates.sort(key=lambda x: x[0], reverse=True)

        top_candidates = candidates[:top_k]

        recalled_tensors = [item[1] for item in top_candidates]
        recall_weights = [item[0] for item in top_candidates]

        return recalled_tensors, recall_weights
