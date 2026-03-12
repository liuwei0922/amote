use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;

use crate::processor::{
    CoreProcessor, CoreProcessorConfig, GraphMemory, MatchOutputProcessor,
    MatchOutputProcessorConfig, StateInputProcessor, StateInputProcessorConfig, TextInputProcessor,
};

#[derive(Config, Debug)]
pub struct RouterConfig {
    pub core_dim: usize,
    pub seq_len: usize,
    pub num_outputs: usize,
}

#[derive(Module, Debug)]
pub struct Router<B: Backend> {
    selector_1: Linear<B>,
    selector_3: Linear<B>,
    arg_gen_1: Linear<B>,
    arg_gen_2: LayerNorm<B>,
}

impl RouterConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Router<B> {
        let input_size = self.seq_len * self.core_dim;
        Router {
            selector_1: LinearConfig::new(input_size, 64).init(device),
            selector_3: LinearConfig::new(64, self.num_outputs).init(device),
            arg_gen_1: LinearConfig::new(input_size, self.core_dim).init(device),
            arg_gen_2: LayerNormConfig::new(self.core_dim).init(device),
        }
    }
}

impl<B: Backend> Router<B> {
    pub fn forward(&self, internal_thoughts: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch_size, seq_len, core_dim] = internal_thoughts.dims();
        let input_size = seq_len * core_dim;

        let flat_input = internal_thoughts.reshape([batch_size, input_size]);

        let s = self.selector_1.forward(flat_input.clone());
        let s = relu(s);
        let route_logits = self.selector_3.forward(s);

        let a = self.arg_gen_1.forward(flat_input);
        let output_arg = self.arg_gen_2.forward(a);

        (route_logits, output_arg)
    }
}

pub struct System<B: Backend> {
    pub dim: usize,
    pub text_proc: TextInputProcessor<B>,
    pub state_proc: StateInputProcessor<B>,
    pub outputs: Vec<MatchOutputProcessor<B>>,
    pub core: CoreProcessor<B>,
    pub router: Router<B>,
    pub memory: GraphMemory<B>,
    pub last_io_pair: Option<(Tensor<B, 3>, Tensor<B, 3>)>,
}

impl<B: Backend> System<B> {
    pub fn new(
        vocab_path: &str,
        model_path: &str,
        device: &B::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dim = 128;
        let seq_len = 2; // TextInput(1) + StateInput(1)

        let text_proc = TextInputProcessor::new(dim, vocab_path, model_path, device.clone())?;
        let state_proc = StateInputProcessorConfig::new(dim).init(device);

        let match_out = MatchOutputProcessorConfig::new(dim).init(device);
        let outputs = vec![match_out];

        let core = CoreProcessorConfig::new(dim).init(device);
        let router = RouterConfig::new(dim, seq_len, outputs.len()).init(device);
        let memory = GraphMemory::new(dim, device);

        Ok(Self {
            dim,
            text_proc,
            state_proc,
            outputs,
            core,
            router,
            memory,
            last_io_pair: None,
        })
    }

    pub fn from_parts(
        text_proc: TextInputProcessor<B>,
        state_proc: StateInputProcessor<B>,
        outputs: Vec<MatchOutputProcessor<B>>,
        core: CoreProcessor<B>,
        router: Router<B>,
        memory: GraphMemory<B>,
    ) -> Self {
        Self {
            dim: memory.dim,
            text_proc,
            state_proc,
            outputs,
            core,
            router,
            memory,
            last_io_pair: None,
        }
    }

    pub fn forward(
        &mut self,
        text_input: &[String],
        state_input: &[String],
        training: bool,
    ) -> (Vec<Option<Tensor<B, 3>>>, Tensor<B, 2>) {
        let t_out = self.text_proc.forward(text_input);
        let s_out = self.state_proc.forward(state_input);

        let mixed_input = Tensor::cat(vec![t_out, s_out], 1);

        let raw_thoughts = self.core.forward(mixed_input.clone());
        let internal_thoughts =
            self.core
                .apply_memory_correction(mixed_input.clone(), raw_thoughts, &self.memory);

        let (route_logits, output_arg) = self.router.forward(internal_thoughts.clone());

        let batch_size = output_arg.dims()[0];
        let output_arg_3d = output_arg.reshape([batch_size, 1, self.dim]);

        self.last_io_pair = Some((mixed_input.detach(), internal_thoughts.detach()));

        let mut results = Vec::new();
        if training {
            for proc in &self.outputs {
                results.push(Some(proc.forward(output_arg_3d.clone())));
            }
        } else {
            let chosen_idx = route_logits
                .clone()
                .slice([0..1])
                .argmax(1)
                .into_scalar()
                .to_i64() as usize;

            for _ in 0..self.outputs.len() {
                results.push(None);
            }
            if let Some(target_proc) = self.outputs.get(chosen_idx) {
                results[chosen_idx] = Some(target_proc.forward(output_arg_3d.clone()));
            }
        }

        (results, route_logits)
    }

    pub fn consolidate_memory(&mut self, correct_mask: &[bool]) {
        if let Some((inputs, outputs)) = self.last_io_pair.take() {
            let [batch_size, seq_len, _] = inputs.dims();
            let dim = self.dim;

            for b in 0..batch_size {
                if !correct_mask[b] {
                    continue;
                }

                for s in 0..seq_len {
                    let inp = inputs
                        .clone()
                        .slice([b..b + 1, s..s + 1, 0..dim])
                        .reshape([dim]);
                    let out = outputs
                        .clone()
                        .slice([b..b + 1, s..s + 1, 0..dim])
                        .reshape([dim]);

                    self.memory.link(inp, out, 1.0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use std::collections::HashMap;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_system_flow() {
        let device = Default::default();
        let dim = 128;
        let seq_len = 2; // Text + State
        let embed_dim = 200;

        let mut vocab = HashMap::new();
        vocab.insert("CMD".to_string(), 0);

        let unk_index = vocab.len();

        let static_embeddings = Tensor::<TestBackend, 2>::random(
            [unk_index + 1, embed_dim],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let text_proc = TextInputProcessor::from_parts(
            vocab,
            static_embeddings,
            unk_index,
            crate::processor::TextTranslatorConfig::new(embed_dim, dim).init(&device),
            device.clone(),
        );

        let state_proc = StateInputProcessorConfig::new(dim).init(&device);
        let outputs = vec![MatchOutputProcessorConfig::new(dim).init(&device)];
        let core = CoreProcessorConfig::new(dim).init(&device);
        let router = RouterConfig::new(dim, seq_len, outputs.len()).init(&device);
        let memory = GraphMemory::<TestBackend>::new(dim, &device);

        let mut system = System::from_parts(text_proc, state_proc, outputs, core, router, memory);

        let text_in = vec!["CMD".to_string()];
        let state_in = vec!["NORTH".to_string()];

        let (results_train, logits) = system.forward(&text_in, &state_in, true);
        println!("Training Logits: {:?}", logits.dims());
        assert!(results_train[0].is_some(), "训练模式下应该触发所有输出");
        assert_eq!(results_train.len(), 1);

        let (results_infer, _) = system.forward(&text_in, &state_in, false);
        assert!(
            results_infer[0].is_some(),
            "推理模式下应该触发得分最高的一路"
        );

        assert_eq!(system.memory.nodes.len(), 0, "巩固前记忆图为空");

        system.consolidate_memory();

        assert!(system.memory.nodes.len() > 0, "记忆图已经被成功填充");
        assert!(system.last_io_pair.is_none(), "记忆缓存应该已经被清空消费");
    }
}
