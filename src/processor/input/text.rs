use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::Int;
use burn::tensor::activation::relu;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

#[derive(Config, Debug)]
pub struct TextTranslatorConfig {
    pub embed_dim: usize,
    pub core_dim: usize,
}

#[derive(Module, Debug)]
pub struct TextTranslator<B: Backend> {
    translator: Linear<B>,
}

impl TextTranslatorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TextTranslator<B> {
        TextTranslator {
            translator: LinearConfig::new(self.embed_dim, self.core_dim).init(device),
        }
    }
}
pub struct TextInputProcessor<B: Backend> {
    vocab: HashMap<String, usize>,
    static_embeddings: Tensor<B, 2>,
    unk_index: usize,
    pub model: TextTranslator<B>,
    device: B::Device,
}

impl<B: Backend> TextInputProcessor<B> {
    pub fn new(
        core_dim: usize,
        vocab_path: &str,
        st_path: &str,
        device: B::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        log::info!("加载 tencent_vocab.json...");
        let file = File::open(vocab_path)?;
        let reader = BufReader::new(file);
        let vocab: HashMap<String, usize> = serde_json::from_reader(reader)?;

        log::info!("加载 tencent_w2v.safetensors...");
        let st_buffer = std::fs::read(st_path)?;
        let st = SafeTensors::deserialize(&st_buffer)?;

        let tensor_view = st.tensor("weight")?;

        let shape = tensor_view.shape();
        let vocab_size = shape[0];
        let embed_dim = shape[1];

        let data_u8 = tensor_view.data();
        let mut data_f32 = vec![0.0f32; data_u8.len() / 4];
        for (i, chunk) in data_u8.chunks_exact(4).enumerate() {
            data_f32[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }

        data_f32.extend(vec![0.0f32; embed_dim]);
        let unk_index = vocab_size;

        let static_embeddings = Tensor::<B, 1>::from_floats(data_f32.as_slice(), &device)
            .reshape([vocab_size + 1, embed_dim]);

        log::info!(
            "初始化 TextTranslator (embed_dim: {}, core_dim: {})...",
            embed_dim,
            core_dim
        );
        let config = TextTranslatorConfig::new(embed_dim, core_dim);
        let model = config.init::<B>(&device);

        Ok(Self {
            vocab,
            static_embeddings,
            unk_index,
            model,
            device,
        })
    }

    pub fn forward(&self, text: &[String]) -> Tensor<B, 3> {
        let batch_size = text.len();
        let embed_dim = self.static_embeddings.dims()[1];
        let mut batch_tensors = Vec::with_capacity(batch_size);

        for s in text {
            let mut char_indices = Vec::new();

            if let Some(&id) = self.vocab.get(s) {
                char_indices.push(id as i32);
            } else {
                for c in s.chars() {
                    let char_str = c.to_string();
                    if let Some(&id) = self.vocab.get(&char_str) {
                        char_indices.push(id as i32);
                    }
                }
            }

            if char_indices.is_empty() {
                char_indices.push(self.unk_index as i32);
            }

            let indices_tensor =
                Tensor::<B, 1, Int>::from_ints(char_indices.as_slice(), &self.device);

            let word_vecs = self.static_embeddings.clone().select(0, indices_tensor);

            let sentence_vec = word_vecs.mean_dim(0);

            batch_tensors.push(sentence_vec);
        }

        let x = Tensor::cat(batch_tensors, 0);

        let x = self.model.translator.forward(x);
        let x = relu(x);

        x.unsqueeze_dim(1)
    }

    // pub fn forward(&self, text: &[String]) -> Tensor<B, 3> {
    //     let mut indices = Vec::with_capacity(text.len());
    //     for s in text {
    //         let id = *self.vocab.get(s).unwrap_or(&self.unk_index);
    //         indices.push(id as i32);
    //     }

    //     let indices_tensor = Tensor::<B, 1, Int>::from_ints(indices.as_slice(), &self.device);

    //     let x = self.static_embeddings.clone().select(0, indices_tensor);

    //     let x = self.model.translator.forward(x);

    //     let x = relu(x);

    //     x.unsqueeze_dim(1)
    // }

    #[allow(dead_code)]
    pub fn from_parts(
        vocab: HashMap<String, usize>,
        static_embeddings: Tensor<B, 2>,
        unk_index: usize,
        model: TextTranslator<B>,
        device: B::Device,
    ) -> Self {
        Self {
            vocab,
            static_embeddings,
            unk_index,
            model,
            device,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_text_input_processor_with_static_embeddings() {
        let device = Default::default();
        let core_dim = 64;
        let embed_dim = 200;

        let mut vocab = HashMap::new();
        vocab.insert("apple".to_string(), 0);
        vocab.insert("banana".to_string(), 1);

        let unk_index = vocab.len();
        let vocab_size_with_unk = vocab.len() + 1;

        let static_embeddings = Tensor::<TestBackend, 2>::random(
            [vocab_size_with_unk, embed_dim],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let config = TextTranslatorConfig::new(embed_dim, core_dim);
        let model = config.init::<TestBackend>(&device);

        let processor =
            TextInputProcessor::from_parts(vocab, static_embeddings, unk_index, model, device);

        let text = vec![
            "apple".to_string(),
            "banana".to_string(),
            "unknown".to_string(),
        ];
        let batch_size = text.len();

        let output = processor.forward(&text);

        assert_eq!(output.dims(), [batch_size, 1, core_dim]);

        println!("Output Tensor dims:\n{:?}", output.dims());
    }
}
