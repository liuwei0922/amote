use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;

#[derive(Config, Debug)]
pub struct StateInputProcessorConfig {
    core_dim: usize,
}

#[derive(Module, Debug)]
pub struct StateInputProcessor<B: Backend> {
    linear: Linear<B>,
}

impl StateInputProcessorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StateInputProcessor<B> {
        StateInputProcessor {
            linear: LinearConfig::new(4, self.core_dim).init(device),
        }
    }
}

impl<B: Backend> StateInputProcessor<B> {
    pub fn forward(&self, states: &[String]) -> Tensor<B, 3> {
        let batch_size = states.len();
        let mut flat_data: Vec<f32> = Vec::with_capacity(batch_size * 4);

        for s in states {
            let mut vec = [0f32; 4];
            match s.as_str() {
                "NORTH" => vec[0] = 1.0,
                "SOUTH" => vec[1] = 1.0,
                "EAST" => vec[2] = 1.0,
                "WEST" => vec[3] = 1.0,
                _ => {}
            }
            flat_data.extend_from_slice(&vec);
        }

        let device = self.devices()[0].clone();

        let raw_tensor =
            Tensor::<B, 1>::from_floats(flat_data.as_slice(), &device).reshape([batch_size, 4]);

        let x = self.linear.forward(raw_tensor);
        let x = relu(x);

        x.unsqueeze_dim(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type TestBackend = NdArray<f32>;

    #[test]
    fn test_state_input_processor_forward() {
        let device = Default::default();

        let core_dim = 8;
        let config = StateInputProcessorConfig::new(core_dim);

        let processor = config.init::<TestBackend>(&device);

        let states = vec![
            "NORTH".to_string(),
            "SOUTH".to_string(),
            "INVALID".to_string(),
        ];
        let batch_size = states.len();

        let output = processor.forward(&states);
        assert_eq!(output.dims(), [batch_size, 1, core_dim]);
        println!("Output Tensor:\n{}", output);
    }
}
