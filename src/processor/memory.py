import torch
import numpy as np


class GraphMemory:
    def __init__(self, dim=128):
        self.nodes = []  # List[Tensor]
        self.adj = {}  # {src_idx: {dst_idx: weight}}
        self.dim = dim

    def _find_similar_nodes(self, tensor, similarity_threshold=0.9):
        if not self.nodes:
            return []

        stack = torch.stack(self.nodes)
        tensor_n = torch.nn.functional.normalize(tensor.unsqueeze(0), p=2, dim=1)
        stack_n = torch.nn.functional.normalize(stack, p=2, dim=1)

        sims = torch.mm(tensor_n, stack_n.t()).squeeze(0)

        indices = (sims > similarity_threshold).nonzero(as_tuple=True)[0]

        results = []
        for idx in indices:
            results.append((idx.item(), sims[idx].item()))

        return results

    def register(self, tensor):
        matches = self._find_similar_nodes(tensor, similarity_threshold=0.99)
        if matches:
            return matches[0][0]

        self.nodes.append(tensor.detach().clone())
        idx = len(self.nodes) - 1
        self.adj[idx] = {}
        return idx

    def link(self, input_tensor, output_tensor, weight=1.0):
        src = self.register(input_tensor)
        dst = self.register(output_tensor)
        current = self.adj[src].get(dst, 0.0)
        self.adj[src][dst] = current + weight

    def query_with_indices(self, input_tensor, threshold=0.1):
        similar_sources = self._find_similar_nodes(
            input_tensor, similarity_threshold=0.9
        )
        if not similar_sources:
            return [], [], []
        candidates = {}

        for src_idx, sim_score in similar_sources:
            if src_idx in self.adj:
                for dst, w in self.adj[src_idx].items():
                    effective_weight = w * sim_score
                    if effective_weight > threshold:
                        candidates[dst] = candidates.get(dst, 0.0) + effective_weight

        idxs = list(candidates.keys())
        tensors = [self.nodes[i] for i in idxs]
        weights = list(candidates.values())

        return idxs, tensors, weights
