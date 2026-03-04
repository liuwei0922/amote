use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Embedding, Linear, Module, VarBuilder, VarMap, embedding, linear};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

pub struct StateInputProcessor {
    linear: Linear,
}

impl StateInputProcessor {
    pub fn new(vs: VarBuilder, core_dim: usize) -> Result<Self> {
        let linear = linear(4, core_dim, vs.pp("translator"))?;
        Ok(Self { linear })
    }

    pub fn forward(&self, states: &[String]) -> Result<Tensor> {
        let batch_size = states.len();
        let mut flat_data = Vec::with_capacity(batch_size * 4);

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

        let device = self.linear.weight().device();
        let raw_tensor = Tensor::from_slice(&flat_data, (batch_size, 4), device)?;

        let x = self.linear.forward(&raw_tensor)?;
        let x = x.relu()?;
        let x = x.unsqueeze(1)?;

        Ok(x)
    }
}

pub struct TextInputProcessor {
    vocab: HashMap<String, usize>,
    embedding_layer: Embedding,
    translator: Linear,
}

impl TextInputProcessor {
    pub fn new(vs: VarBuilder, core_dim: usize, vocab_path: &str, st_path: &str) -> Result<Self> {
        log::info!("加载词表...");
        let file = File::open(vocab_path)?;
        let reader = BufReader::new(file);
        let vocab: HashMap<String, usize> = serde_json::from_reader(reader)?;

        log::info!("加载模型参数...");
        let st_vs = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[st_path],
                candle_core::DType::F32,
                vs.device(),
            )?
        };

        let vocab_size = vocab.len();
        let embed_dim = 200;

        let embedding_layer = embedding(vocab_size, embed_dim, st_vs)?;

        let translator = linear(embed_dim, core_dim, vs.pp("translator"))?;

        Ok(Self {
            vocab,
            embedding_layer,
            translator,
        })
    }

    pub fn forward(&self, text: &[String]) -> Result<Tensor> {
        let device = self.embedding_layer.embeddings().device();

        let mut indices = Vec::with_capacity(text.len());
        for s in text {
            let id = *self.vocab.get(s).unwrap_or(&0);
            indices.push(id as u32);
        }

        let input_tensor = Tensor::from_slice(&indices, (text.len(),), device)?;

        let x = self.embedding_layer.forward(&input_tensor)?;
        let x = self.translator.forward(&x.detach())?;
        let x = x.relu()?;
        Ok(x.unsqueeze(1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;
    use std::fs;

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
        let vocab = HashMap::from([
            ("测试".to_string(), 0),
            ("Rust".to_string(), 1),
            ("Candle".to_string(), 2),
        ]);
        let len = vocab.len();
        let file = File::create(path)?;
        serde_json::to_writer(file, &vocab)?;
        Ok(len)
    }

    #[test]
    fn test_text_processor_flow() -> Result<()> {
        let vocab_path = "test_temp_vocab.json";
        let model_path = "test_temp_model.safetensors";

        let vocab_size = create_dummy_vocab(vocab_path)?;

        create_dummy_safetensors(model_path, "weight", vocab_size, 200)?;

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let core_dim = 64;

        let model = TextInputProcessor::new(vs, core_dim, vocab_path, model_path)
            .expect("模型加载挂了！请看上面的报错信息");

        let inputs = vec!["Rust".to_string(), "未知词".to_string()];

        let output = model.forward(&inputs).expect("Forward 计算挂了");

        println!("Text Output Shape: {:?}", output.shape());
        assert_eq!(output.dims(), &[2, 1, 64]);

        let _ = fs::remove_file(vocab_path);
        let _ = fs::remove_file(model_path);

        Ok(())
    }

    #[test]
    fn test_state_processor_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = StateInputProcessor::new(vs, 128)?;

        let inputs = vec!["NORTH".to_string(), "SOUTH".to_string()];
        let output = model.forward(&inputs)?;

        assert_eq!(output.dims(), &[2, 1, 128]);
        Ok(())
    }
}
