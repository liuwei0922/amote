use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};

pub struct MatchOutputProcessor {
    decoder: Linear,
}

impl MatchOutputProcessor {
    pub fn new(vs: VarBuilder, core_dim: usize) -> Result<Self> {
        let decoder = linear(core_dim, 2, vs.pp("decoder"))?;
        Ok(Self { decoder })
    }

    pub fn forward(&self, concept: &Tensor) -> Result<Tensor> {
        Ok(self.decoder.forward(concept)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_match_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let core_dim = 128;
        let model = MatchOutputProcessor::new(vs, core_dim)?;

        let input = Tensor::randn(0f32, 1f32, (3, 1, core_dim), &device)?;
        let output = model.forward(&input)?;
        println!("Output Shape: {:?}", output.shape());

        assert_eq!(output.dims(), &[3, 1, 2]);
        Ok(())
    }
}
