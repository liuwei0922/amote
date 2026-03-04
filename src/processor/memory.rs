use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use std::collections::HashMap;

pub struct GraphMemory {
    pub nodes: Vec<Tensor>,
    pub adj: HashMap<usize, HashMap<usize, f32>>,
    pub dim: usize,
}

impl GraphMemory {
    pub fn new(dim: usize) -> Self {
        Self {
            nodes: Vec::new(),
            adj: HashMap::new(),
            dim,
        }
    }

    pub fn find_similar_nodes(
        &self,
        tensor: &Tensor,
        similarity_threshold: f32,
    ) -> Result<Vec<(usize, f32)>> {
        if self.nodes.is_empty() {
            return Ok(vec![]);
        }

        let stack = Tensor::stack(&self.nodes, 0)?;
        let tensor_n = tensor
            .unsqueeze(0)?
            .broadcast_div(&tensor.sqr()?.sum_keepdim(0)?.sqrt()?)?;

        let t_norm = tensor
            .unsqueeze(0)?
            .broadcast_div(&tensor.sqr()?.sum_keepdim(0)?.sqrt()?)?;
        let s_norm = stack.broadcast_div(&stack.sqr()?.sum_keepdim(1)?.sqrt()?)?;

        let sims = t_norm.matmul(&s_norm.t()?)?.squeeze(0)?;

        let sims_vec: Vec<f32> = sims.to_vec1()?;

        let mut results = Vec::new();
        for (idx, &score) in sims_vec.iter().enumerate() {
            if score > similarity_threshold {
                results.push((idx, score));
            }
        }

        Ok(results)
    }

    pub fn register(&mut self, tensor: &Tensor) -> Result<usize> {
        let matches = self.find_similar_nodes(tensor, 0.95)?;

        if let Some((idx, _)) = matches.first() {
            return Ok(*idx);
        }

        self.nodes.push(tensor.clone());
        let idx = self.nodes.len() - 1;
        self.adj.insert(idx, HashMap::new());

        Ok(idx)
    }

    pub fn link(
        &mut self,
        input_tensor: &Tensor,
        output_tensor: &Tensor,
        weight: f32,
    ) -> Result<()> {
        let src = self.register(input_tensor)?;
        let dst = self.register(output_tensor)?;

        let src_adj = self.adj.get_mut(&src).unwrap();
        let current = *src_adj.get(&dst).unwrap_or(&0.0);
        src_adj.insert(dst, current + weight);

        Ok(())
    }

    pub fn query_with_indices(
        &self,
        input_tensor: &Tensor,
        threshold: f32,
    ) -> Result<(Vec<usize>, Vec<Tensor>, Vec<f32>)> {
        let similar_sources = self.find_similar_nodes(input_tensor, 0.9)?;

        if similar_sources.is_empty() {
            return Ok((vec![], vec![], vec![]));
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

        Ok((idxs, tensors, weights))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_logic() -> Result<()> {
        let device = Device::Cpu;
        let mut mem = GraphMemory::new(4);

        let t1 = Tensor::from_slice(&[1.0f32, 0., 0., 0.], (4,), &device)?;
        let t2 = Tensor::from_slice(&[0.0f32, 1., 0., 0.], (4,), &device)?;
        let t3 = Tensor::from_slice(&[0.99f32, 0.01, 0., 0.], (4,), &device)?;

        let idx1 = mem.register(&t1)?;
        assert_eq!(idx1, 0);

        let idx3 = mem.register(&t3)?;
        assert_eq!(idx1, idx3, "t3 应该极其相似 t1，返回相同 ID");

        mem.link(&t1, &t2, 1.0)?;

        let (idxs, tensors, weights) = mem.query_with_indices(&t1, 0.1)?;
        assert!(!idxs.is_empty());
        assert_eq!(idxs[0], 1);
        println!("Query result: indices={:?}, weights={:?}", idxs, weights);

        Ok(())
    }
}
