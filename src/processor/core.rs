use anyhow::Result;
use candle_core::{Device, IndexOp, Module, Tensor};
use candle_nn::{LayerNorm, Linear, Sequential, VarBuilder, layer_norm, linear};
use std::cell::RefCell;
use std::collections::HashMap;

use crate::processor::GraphMemory;

pub struct CoreProcessor {
    dim: usize,
    memory: RefCell<GraphMemory>,
    fusion_net: Sequential,
    op_net: Linear,
    last_io_pair: RefCell<Option<(Tensor, Tensor)>>,
}

impl CoreProcessor {
    pub fn new(vs: VarBuilder, core_dim: usize) -> Result<Self> {
        let memory = GraphMemory::new(core_dim);

        let fusion_net = candle_nn::seq()
            .add(linear(core_dim, core_dim, vs.pp("fusion_net.0"))?)
            .add(layer_norm(core_dim, 1e-5, vs.pp("fusion_net.1"))?)
            .add_fn(|x| x.relu());

        let op_net = linear(core_dim, core_dim, vs.pp("op_net"))?;

        Ok(Self {
            dim: core_dim,
            memory: RefCell::new(memory),
            fusion_net,
            op_net,
            last_io_pair: RefCell::new(None),
        })
    }

    pub fn forward(&self, input_tensor: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _dim) = input_tensor.dims3()?;
        let device = input_tensor.device();

        let features = self.fusion_net.forward(input_tensor)?;
        let raw_output = self.op_net.forward(&features)?;
        let mut corrected_output_list = Vec::with_capacity(batch_size);

        let memory = self.memory.borrow();

        for b in 0..batch_size {
            struct CandidateData {
                tensor: Tensor,
                weights: Vec<f32>,
            }
            let mut candidates: HashMap<usize, CandidateData> = HashMap::new();

            for s in 0..seq_len {
                let token = input_tensor.i((b, s))?;

                let (idxs, tensors, weights) = memory.query_with_indices(&token, 0.1)?;

                for ((idx, tens), w) in idxs
                    .into_iter()
                    .zip(tensors.into_iter())
                    .zip(weights.into_iter())
                {
                    let entry = candidates.entry(idx).or_insert_with(|| CandidateData {
                        tensor: tens,
                        weights: vec![0.0; seq_len],
                    });
                    entry.weights[s] = w;
                }
            }

            let mut final_seq = Vec::with_capacity(seq_len);

            for s_out in 0..seq_len {
                let target_vec = raw_output.i((b, s_out))?;
                let mut correction = Tensor::zeros_like(&target_vec)?;
                let mut total_influence = 0.0f32;

                for data in candidates.values() {
                    let w_list = &data.weights;
                    let mem_vec = &data.tensor;

                    let compound_weight: f32 = w_list.iter().sum();

                    if compound_weight > 0.01 {
                        let norm_sq = (mem_vec * mem_vec)?.sum_all()?.to_scalar::<f32>()?;

                        if norm_sq > 1e-6 {
                            let dot = (target_vec.clone() * mem_vec)?
                                .sum_all()?
                                .to_scalar::<f32>()?;

                            let scale = (dot / norm_sq) * compound_weight;
                            let proj = (mem_vec * scale as f64)?;

                            correction = (correction + proj)?;
                            total_influence += compound_weight;
                        }
                    }
                }

                let final_vec = if total_influence > 0.01 {
                    let scale = 1.0 / (total_influence + 1e-5);
                    let correction = (correction * scale as f64)?;
                    (&target_vec + (correction * 0.5)?)?
                } else {
                    target_vec
                };

                final_seq.push(final_vec);
            }

            corrected_output_list.push(Tensor::stack(&final_seq, 0)?);
        }

        let final_output = Tensor::stack(&corrected_output_list, 0)?;

        *self.last_io_pair.borrow_mut() = Some((input_tensor.detach(), final_output.detach()));

        Ok(final_output)
    }

    pub fn update_memory(&self, mask: Option<&Tensor>) -> Result<()> {
        let pair_ref = self.last_io_pair.borrow();
        if let Some((inputs, outputs)) = pair_ref.as_ref() {
            let (batch_size, seq_len, _) = inputs.dims3()?;

            let mut mem = self.memory.borrow_mut();

            for b in 0..batch_size {
                let should_update = match mask {
                    Some(m) => m.i(b)?.to_scalar::<u8>()? != 0,
                    None => true,
                };

                if should_update {
                    for s in 0..seq_len {
                        let inp = inputs.i((b, s))?;
                        let out = outputs.i((b, s))?;
                        mem.link(&inp, &out, 1.0)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_core_processor_flow() -> Result<()> {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let core_dim = 16;
        let model = CoreProcessor::new(vs, core_dim)?;

        let input = Tensor::randn(0f32, 1f32, (2, 5, core_dim), &device)?;

        let out1 = model.forward(&input)?;
        assert_eq!(out1.dims(), &[2, 5, core_dim]);
        println!("Forward 1 done.");

        model.update_memory(None)?;

        {
            let mem = model.memory.borrow();
            assert!(!mem.nodes.is_empty(), "Memory 应该记录了节点");
            println!("Memory nodes count: {}", mem.nodes.len());
        }

        let noisy_input = (input + 0.01)?;
        let out2 = model.forward(&noisy_input)?;
        assert_eq!(out2.dims(), &[2, 5, core_dim]);
        println!("Forward 2 done.");

        Ok(())
    }
}
