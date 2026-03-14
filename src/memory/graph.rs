use burn::prelude::*;

pub struct GraphMemory<B: Backend> {
    pub dim: usize,
    pub device: B::Device,
    pub nodes: Vec<Tensor<B, 1>>,
    pub node_weights: Vec<f32>,

    cached_bank: Option<Tensor<B, 2>>,
}

impl<B: Backend> GraphMemory<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            dim,
            device: device.clone(),
            nodes: Vec::new(),
            node_weights: Vec::new(),
            cached_bank: None,
        }
    }

    fn sync_bank(&mut self) {
        if self.nodes.is_empty() {
            self.cached_bank = None;
            return;
        }

        let nodes_2d: Vec<Tensor<B, 2>> = self
            .nodes
            .iter()
            .map(|t| t.clone().reshape([1, self.dim]))
            .collect();

        self.cached_bank = Some(Tensor::cat(nodes_2d, 0).detach());
    }

    pub fn query_nearest(
        &self,
        v_query: &Tensor<B, 2>,
    ) -> Option<(Tensor<B, 2>, Vec<f32>, Vec<usize>)> {
        let bank = self.cached_bank.as_ref()?;
        let [batch_size, dim] = v_query.dims();

        let v_sq = (v_query.clone() * v_query.clone()).sum_dim(1);
        let q_norm = v_query.clone() / (v_sq.sqrt() + 1e-8);

        let sims = q_norm.matmul(bank.clone().transpose());

        let max_sims = sims.clone().max_dim(1);
        let max_indices = sims.argmax(1);

        let max_sims_vec = max_sims.into_data().to_vec::<f32>().unwrap();

        let max_indices_vec: Vec<usize> = max_indices
            .into_data()
            .to_vec::<i64>()
            .unwrap()
            .into_iter()
            .map(|i| i as usize)
            .collect();

        let mut nearest_tensors = Vec::with_capacity(batch_size);
        for &idx in &max_indices_vec {
            let nearest = bank.clone().slice([idx..idx + 1, 0..dim]).reshape([1, dim]);
            nearest_tensors.push(nearest);
        }

        Some((
            Tensor::cat(nearest_tensors, 0),
            max_sims_vec,
            max_indices_vec,
        ))
    }

    pub fn get_collision_avoidance_vector(
        &self,
        v_candidate: &Tensor<B, 2>,
        threshold: f32,
    ) -> Option<Tensor<B, 2>> {
        if let Some((nearest_vecs, sims, _)) = self.query_nearest(v_candidate) {
            let [batch_size, _] = v_candidate.dims();
            let mut avoid_batch = Vec::with_capacity(batch_size);
            let mut triggered = false;

            for b in 0..batch_size {
                if sims[b] > threshold {
                    let v_collide = nearest_vecs.clone().slice([b..b + 1, 0..self.dim]);
                    avoid_batch.push(v_collide);
                    triggered = true;
                } else {
                    avoid_batch.push(Tensor::<B, 2>::zeros([1, self.dim], &self.device));
                }
            }

            if triggered {
                return Some(Tensor::cat(avoid_batch, 0));
            }
        }
        None
    }

    pub fn register(&mut self, v_new: Tensor<B, 1>) -> usize {
        let v_sq = (v_new.clone() * v_new.clone()).sum().sqrt();
        let v_norm = v_new.clone() / (v_sq + 1e-8);
        let v_norm_2d = v_norm.clone().reshape([1, self.dim]);

        if let Some((_nearest_2d, sims, indices)) = self.query_nearest(&v_norm_2d) {
            let sim = sims[0];
            let nearest_idx = indices[0];

            if sim > 0.95 {
                let old_weight = self.node_weights[nearest_idx];
                let old_tensor = self.nodes[nearest_idx].clone();

                let updated = (old_tensor * old_weight + v_norm.detach()) / (old_weight + 1.0);

                let u_sq = (updated.clone() * updated.clone()).sum().sqrt();
                self.nodes[nearest_idx] = updated / (u_sq + 1e-8);
                self.node_weights[nearest_idx] = (old_weight + 1.0).min(100.0);

                self.sync_bank();
                return nearest_idx;
            }
        }

        let new_idx = self.nodes.len();
        self.nodes.push(v_norm.detach());
        self.node_weights.push(1.0);

        self.sync_bank();
        new_idx
    }
}
