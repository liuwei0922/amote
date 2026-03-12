use crate::processor::GraphMemory;

use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;

#[derive(Config, Debug)]
pub struct CoreProcessorConfig {
    pub core_dim: usize,
}

#[derive(Module, Debug)]
pub struct CoreProcessor<B: Backend> {
    fusion_linear: Linear<B>,
    fusion_norm: LayerNorm<B>,
    op_net: Linear<B>,
}

impl CoreProcessorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CoreProcessor<B> {
        CoreProcessor {
            fusion_linear: LinearConfig::new(self.core_dim, self.core_dim).init(device),
            fusion_norm: LayerNormConfig::new(self.core_dim).init(device),
            op_net: LinearConfig::new(self.core_dim, self.core_dim).init(device),
        }
    }
}

impl<B: Backend> CoreProcessor<B> {
    pub fn forward(&self, input_tensor: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fusion_linear.forward(input_tensor);
        let x = self.fusion_norm.forward(x);
        let features = relu(x);
        self.op_net.forward(features)
    }

    pub fn apply_memory_correction(
        &self,
        input_tensor: Tensor<B, 3>,
        raw_output: Tensor<B, 3>,
        memory: &GraphMemory<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _dim] = input_tensor.dims();
        let device = &memory.device;
        let core_dim = memory.dim;

        let mut corrected_batches = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let mut final_seq = Vec::with_capacity(seq_len);

            for s in 0..seq_len {
                let token_in = input_tensor
                    .clone()
                    .slice([b..b + 1, s..s + 1, 0..core_dim])
                    .reshape([core_dim]);

                let target_vec = raw_output
                    .clone()
                    .slice([b..b + 1, s..s + 1, 0..core_dim])
                    .reshape([core_dim]);

                let (_, mem_tensors, weights) = memory.query_with_indices(&token_in, 0.1);

                let mut correction = Tensor::<B, 1>::zeros([core_dim], device);
                let mut total_influence = 0.0f32;

                for (mem_vec, w) in mem_tensors.into_iter().zip(weights.into_iter()) {
                    if w > 0.01 {
                        let norm_sq = (mem_vec.clone() * mem_vec.clone()).sum();
                        let norm_sq_val = norm_sq.clone().into_scalar().to_f32();

                        if norm_sq_val > 1e-6 {
                            let dot = (target_vec.clone() * mem_vec.clone()).sum();
                            let proj = mem_vec * (dot / norm_sq);

                            correction = correction + (proj * w);
                            total_influence += w;
                        }
                    }
                }

                let final_vec = if total_influence > 0.01 {
                    let correction_norm = correction / (total_influence + 1e-5);
                    target_vec + (correction_norm * 0.5)
                } else {
                    target_vec
                };

                final_seq.push(final_vec.reshape([1, 1, core_dim]));
            }

            let batch_tensor = Tensor::cat(final_seq, 1);
            corrected_batches.push(batch_tensor);
        }

        Tensor::cat(corrected_batches, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_core_processor_and_memory_integration() {
        let device = Default::default();
        let core_dim = 64;
        let batch_size = 2;
        let seq_len = 3;

        let mut memory = GraphMemory::<TestBackend>::new(core_dim, &device);
        let config = CoreProcessorConfig::new(core_dim);
        let processor = config.init::<TestBackend>(&device);

        let input_tensor = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, core_dim],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let raw_output = processor.forward(input_tensor.clone());

        let final_output =
            processor.apply_memory_correction(input_tensor.clone(), raw_output.clone(), &memory);

        assert_eq!(raw_output.dims(), [batch_size, seq_len, core_dim]);
        assert_eq!(final_output.dims(), [batch_size, seq_len, core_dim]);
        assert_eq!(memory.nodes.len(), 0);

        for b in 0..batch_size {
            for s in 0..seq_len {
                let inp_vec = input_tensor
                    .clone()
                    .slice([b..b + 1, s..s + 1, 0..core_dim])
                    .reshape([core_dim]);

                let out_vec = final_output
                    .clone()
                    .slice([b..b + 1, s..s + 1, 0..core_dim])
                    .reshape([core_dim]);

                memory.link(inp_vec, out_vec, 1.0);
            }
        }

        assert!(memory.nodes.len() > 0);
        assert!(memory.node_weights.len() > 0);

        let raw_output_2 = processor.forward(input_tensor.clone());
        let final_output_2 =
            processor.apply_memory_correction(input_tensor.clone(), raw_output_2.clone(), &memory);

        assert_eq!(final_output_2.dims(), [batch_size, seq_len, core_dim]);

        println!("当前图记忆节点数量: {}", memory.nodes.len());
        println!("第一个节点的熟悉度(权重): {}", memory.node_weights[0]);
    }
}
