use burn::prelude::*;
use std::collections::HashMap;

pub struct GraphMemory<B: Backend> {
    pub nodes: Vec<Tensor<B, 1>>,
    pub node_weights: Vec<f32>,
    pub adj: HashMap<usize, HashMap<usize, f32>>,
    pub dim: usize,
    pub device: B::Device,
}

impl<B: Backend> GraphMemory<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            nodes: Vec::new(),
            node_weights: Vec::new(),
            adj: HashMap::new(),
            dim,
            device: device.clone(),
        }
    }

    fn find_similar_nodes(&self, tensor: &Tensor<B, 1>, threshold: f32) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() {
            return vec![];
        }

        let n_nodes = self.nodes.len();
        let dim = self.dim;

        let nodes_2d: Vec<Tensor<B, 2>> = self
            .nodes
            .iter()
            .map(|t| t.clone().reshape([1, dim]))
            .collect();
        let stack = Tensor::cat(nodes_2d, 0);

        let t_2d = tensor.clone().reshape([1, dim]);

        let tensor_norm = (t_2d.clone() * t_2d.clone()).sum_dim(1).sqrt();

        let t_n = t_2d / tensor_norm;

        let stack_norm = (stack.clone() * stack.clone()).sum_dim(1).sqrt();
        let s_n = stack / stack_norm;

        let sims = t_n.matmul(s_n.transpose()).reshape([n_nodes]);

        let sims_vec = sims.into_data().to_vec::<f32>().unwrap();

        let mut results = Vec::new();
        for (idx, &score) in sims_vec.iter().enumerate() {
            if score > threshold {
                results.push((idx, score));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    pub fn register(&mut self, tensor: Tensor<B, 1>) -> usize {
        let matches = self.find_similar_nodes(&tensor, 0.95);

        if let Some((idx, _sim)) = matches.first() {
            let old_weight = self.node_weights[*idx];
            let old_tensor = self.nodes[*idx].clone();
            let new_tensor = tensor.detach();

            let updated_tensor = (old_tensor * old_weight + new_tensor) / (old_weight + 1.0);

            self.nodes[*idx] = updated_tensor;

            let w = self.node_weights[*idx];
            self.node_weights[*idx] = (w + 1.0).min(100.0);

            return *idx;
        }

        let new_idx = self.nodes.len();
        self.nodes.push(tensor.detach());
        self.node_weights.push(1.0);
        self.adj.insert(new_idx, HashMap::new());

        new_idx
    }

    pub fn link(&mut self, input_tensor: Tensor<B, 1>, output_tensor: Tensor<B, 1>, weight: f32) {
        let src = self.register(input_tensor);
        let dst = self.register(output_tensor);

        let src_adj = self.adj.get_mut(&src).unwrap();
        let current = *src_adj.get(&dst).unwrap_or(&0.0);
        src_adj.insert(dst, current + weight);
    }

    pub fn query_with_indices(
        &self,
        input_tensor: &Tensor<B, 1>,
        threshold: f32,
    ) -> (Vec<usize>, Vec<Tensor<B, 1>>, Vec<f32>) {
        let similar_sources = self.find_similar_nodes(input_tensor, 0.9);
        if similar_sources.is_empty() {
            return (vec![], vec![], vec![]);
        }

        let mut candidates_map: HashMap<usize, f32> = HashMap::new();

        for (src_idx, sim_score) in similar_sources {
            if let Some(neighbors) = self.adj.get(&src_idx) {
                for (&dst, &w) in neighbors {
                    let effective_weight = w * sim_score;
                    if effective_weight > threshold {
                        *candidates_map.entry(dst).or_default() += effective_weight;
                    }
                }
            }
        }

        let mut idxs = Vec::new();
        let mut tensors = Vec::new();
        let mut weights = Vec::new();

        for (idx, w) in candidates_map {
            idxs.push(idx);
            tensors.push(self.nodes[idx].clone());
            weights.push(w);
        }

        (idxs, tensors, weights)
    }
}
