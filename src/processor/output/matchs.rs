use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

#[derive(Config, Debug)]
pub struct MatchOutputProcessorConfig {
    pub core_dim: usize,
}

#[derive(Module, Debug)]
pub struct MatchOutputProcessor<B: Backend> {
    decoder: Linear<B>,
}

impl MatchOutputProcessorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MatchOutputProcessor<B> {
        MatchOutputProcessor {
            decoder: LinearConfig::new(self.core_dim, 2).init(device),
        }
    }
}

impl<B: Backend> MatchOutputProcessor<B> {
    pub fn forward(&self, concept: Tensor<B, 3>) -> Tensor<B, 3> {
        self.decoder.forward(concept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type TestBackend = NdArray<f32>;

    #[test]
    fn test_match_output_shape() {
        let device = Default::default();
        let core_dim = 128;

        let config = MatchOutputProcessorConfig::new(core_dim);
        let model = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random(
            [3, 1, core_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = model.forward(input);

        println!("Output Shape: {:?}", output.dims());
        assert_eq!(output.dims(), [3, 1, 2]);
    }
}
