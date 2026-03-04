use anyhow::Result;
use candle_core::{Device, IndexOp, Module, Tensor};
use candle_nn::{LayerNorm, Linear, Sequential, VarBuilder, layer_norm, linear};

use crate::processor::{
    CoreProcessor, MatchOutputProcessor, StateInputProcessor, TextInputProcessor,
};

pub struct Router {
    selector: Sequential,
    arg_generator: Sequential,
}

impl Router {
    pub fn new(
        vs: VarBuilder,
        core_dim: usize,
        seq_len: usize,
        num_outputs: usize,
    ) -> Result<Self> {
        let input_size = seq_len * core_dim;

        let selector = candle_nn::seq()
            .add(linear(input_size, 64, vs.pp("selector.1"))?)
            .add_fn(|x| x.relu())
            .add(linear(64, num_outputs, vs.pp("selector.3"))?);

        let arg_generator = candle_nn::seq()
            .add(linear(input_size, core_dim, vs.pp("arg_generator.1"))?)
            .add(layer_norm(core_dim, 1e-5, vs.pp("arg_generator.2"))?);

        Ok(Self {
            selector,
            arg_generator,
        })
    }

    pub fn forward(&self, internal_thoughts: &Tensor) -> Result<(Tensor, Tensor)> {
        let flat_input = internal_thoughts.flatten_from(1)?;

        let selection_logits = self.selector.forward(&flat_input)?;
        let output_arg = self.arg_generator.forward(&flat_input)?;

        Ok((selection_logits, output_arg))
    }
}

pub struct System {
    dim: usize,
    text_proc: TextInputProcessor,
    state_proc: StateInputProcessor,
    outputs: Vec<MatchOutputProcessor>,
    core: CoreProcessor,
    router: Router,
}

impl System {
    pub fn new(vs: VarBuilder, vocab_path: &str, model_path: &str) -> Result<Self> {
        let dim = 128;
        let seq_len = 2;

        let text_proc = TextInputProcessor::new(vs.pp("inputs.0"), dim, vocab_path, model_path)?;
        let state_proc = StateInputProcessor::new(vs.pp("inputs.1"), dim)?;

        let match_out = MatchOutputProcessor::new(vs.pp("outputs.0"), dim)?;
        let outputs = vec![match_out];

        let core = CoreProcessor::new(vs.pp("core"), dim)?;

        let router = Router::new(vs.pp("router"), dim, seq_len, outputs.len())?;

        Ok(Self {
            dim,
            text_proc,
            state_proc,
            outputs,
            core,
            router,
        })
    }

    pub fn forward(
        &self,
        text_input: &[String],
        state_input: &[String],
        training: bool,
    ) -> Result<(Vec<Option<Tensor>>, Tensor)> {
        let t_out = self.text_proc.forward(text_input)?;
        let s_out = self.state_proc.forward(state_input)?;

        let mixed_input = Tensor::cat(&[&t_out, &s_out], 1)?;
        let internal_thoughts = self.core.forward(&mixed_input)?;
        let (route_logits, output_arg) = self.router.forward(&internal_thoughts)?;

        let mut results = Vec::new();

        if training {
            for proc in &self.outputs {
                let res = proc.forward(&output_arg)?;
                results.push(Some(res));
            }
        } else {
            let chosen_idx = route_logits.i(0)?.argmax(0)?.to_scalar::<u32>()? as usize;

            for _ in 0..self.outputs.len() {
                results.push(None);
            }

            if let Some(target_proc) = self.outputs.get(chosen_idx) {
                let res = target_proc.forward(&output_arg)?;
                results[chosen_idx] = Some(res);
            }
        }

        Ok((results, route_logits))
    }

    pub fn consolidate_memory(&self, correct_mask: Option<&Tensor>) -> Result<()> {
        self.core.update_memory(correct_mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;
    use std::fs;
    use std::fs::File;

    fn create_dummy_safetensors(
        path: &str,
        key_name: &str,
        vocab_size: usize,
        dim: usize,
    ) -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::randn(0f32, 1f32, (vocab_size, dim), &device)?;
        let map = HashMap::from([(key_name.to_string(), weight)]);
        candle_core::safetensors::save(&map, path)?;
        Ok(())
    }

    fn create_dummy_vocab(path: &str) -> Result<usize> {
        let vocab = HashMap::from([("CMD".to_string(), 0), ("ARG".to_string(), 1)]);
        let len = vocab.len();
        let file = File::create(path)?;
        serde_json::to_writer(file, &vocab)?;
        Ok(len)
    }

    #[test]
    fn test_system_flow() -> Result<()> {
        let vocab_path = "sys_temp_vocab.json";
        let model_path = "sys_temp_model.safetensors";
        let vocab_size = create_dummy_vocab(vocab_path)?;
        create_dummy_safetensors(model_path, "weight", vocab_size, 200)?;

        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let system = System::new(vs, vocab_path, model_path)?;

        let text_in = vec!["CMD".to_string()];
        let state_in = vec!["NORTH".to_string()];

        let (results_train, logits) = system.forward(&text_in, &state_in, true)?;
        println!("Training Logits: {:?}", logits);
        assert!(results_train[0].is_some());
        assert_eq!(results_train.len(), 1);

        let (results_infer, _) = system.forward(&text_in, &state_in, false)?;
        assert!(results_infer[0].is_some());

        system.consolidate_memory(None)?;

        let _ = fs::remove_file(vocab_path);
        let _ = fs::remove_file(model_path);

        Ok(())
    }
}
