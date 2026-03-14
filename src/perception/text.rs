use burn::nn::{Embedding, EmbeddingConfig};
use burn::prelude::*;
use burn::tensor::Int;
use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Debug)]
pub struct CharTokenizer {
    pub char_to_id: HashMap<char, usize>,
    pub current_size: usize,
    pub max_capacity: usize,
}

impl CharTokenizer {
    pub fn new(max_capacity: usize) -> Self {
        let mut char_to_id = HashMap::new();
        char_to_id.insert('\0', 0);
        Self {
            char_to_id,
            current_size: 1,
            max_capacity,
        }
    }

    pub fn get_or_add(&mut self, c: char) -> usize {
        if let Some(&id) = self.char_to_id.get(&c) {
            id
        } else {
            if self.current_size < self.max_capacity {
                let new_id = self.current_size;
                self.char_to_id.insert(c, new_id);
                self.current_size += 1;
                new_id
            } else {
                0
            }
        }
    }
}

#[derive(Config, Debug)]
pub struct TextEncoderConfig {
    pub max_vocab_size: usize,
    pub core_dim: usize,
}

#[derive(Module, Debug)]
pub struct TextEncoder<B: Backend> {
    pub embedding: Embedding<B>,
}

impl TextEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TextEncoder<B> {
        TextEncoder {
            embedding: EmbeddingConfig::new(self.max_vocab_size, self.core_dim).init(device),
        }
    }
}

pub struct TextInputProcessor<B: Backend> {
    tokenizer: RwLock<CharTokenizer>,
    pub encoder: TextEncoder<B>,
    device: B::Device,
}

impl<B: Backend> TextInputProcessor<B> {
    pub fn new(core_dim: usize, max_capacity: usize, device: &B::Device) -> Self {
        let tokenizer = RwLock::new(CharTokenizer::new(max_capacity));
        let encoder = TextEncoderConfig::new(max_capacity, core_dim).init(device);

        Self {
            tokenizer,
            encoder,
            device: device.clone(),
        }
    }

    pub fn forward(&self, texts: &[String]) -> Tensor<B, 3> {
        let batch_size = texts.len();

        let mut batch_indices: Vec<Vec<i32>> = Vec::with_capacity(batch_size);
        let mut max_seq_len = 1;

        {
            let mut tokenizer = self.tokenizer.write().unwrap();
            for text in texts {
                let mut indices = Vec::new();
                for c in text.chars() {
                    if !c.is_whitespace() {
                        let id = tokenizer.get_or_add(c);
                        indices.push(id as i32);
                    }
                }
                if indices.is_empty() {
                    indices.push(0);
                }
                if indices.len() > max_seq_len {
                    max_seq_len = indices.len();
                }
                batch_indices.push(indices);
            }
        }

        let mut flat_padded_indices = Vec::with_capacity(batch_size * max_seq_len);
        for mut indices in batch_indices {
            indices.resize(max_seq_len, 0);
            flat_padded_indices.extend(indices);
        }

        let indices_tensor =
            Tensor::<B, 1, Int>::from_ints(flat_padded_indices.as_slice(), &self.device)
                .reshape([batch_size, max_seq_len]);

        self.encoder.embedding.forward(indices_tensor)
    }
}
