use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

#[derive(Config, Debug)]
pub struct GeneratorConfig {
    pub core_dim: usize,
}

#[derive(Module, Debug)]
pub struct Generator<B: Backend> {
    proj: Linear<B>,
}

impl GeneratorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Generator<B> {
        Generator {
            proj: LinearConfig::new(self.core_dim, self.core_dim)
                .with_bias(false)
                .init(device),
        }
    }
}

impl<B: Backend> Generator<B> {
    pub fn forward(
        &self,
        v_inst: Tensor<B, 2>,
        memory: &crate::memory::GraphMemory<B>,
    ) -> Tensor<B, 2> {
        let mut v_op = self.proj.forward(v_inst);

        if let Some(v_mem) = memory.get_collision_avoidance_vector(&v_op, 0.85) {
            let dot = (v_op.clone() * v_mem.clone()).sum_dim(1);
            let proj = v_mem * dot;

            v_op = v_op - proj;
        }

        let v_sq = (v_op.clone() * v_op.clone()).sum_dim(1);
        let v_norm = v_op / (v_sq.sqrt() + 1e-8);

        v_norm
    }
}
