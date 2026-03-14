use crate::algebra::generator::{Generator, GeneratorConfig};
use crate::algebra::operator::Operator;
use crate::memory::graph::GraphMemory;
use crate::perception::text::TextInputProcessor;
use crate::perception::vision::{VisionEncoder, VisionEncoderConfig};
use burn::prelude::*;

pub struct AgentSystem<B: Backend> {
    pub text_proc: TextInputProcessor<B>,
    pub vision_encoder: VisionEncoder<B>,
    pub generator: Generator<B>,
    pub v_true: Tensor<B, 1>,
    pub v_false: Tensor<B, 1>,
}

impl<B: Backend> AgentSystem<B> {
    pub fn new(text_proc: TextInputProcessor<B>, core_dim: usize, device: &B::Device) -> Self {
        let vision_encoder = VisionEncoderConfig::new(core_dim).init(device);
        let generator = GeneratorConfig::new(core_dim).init(device);
        let v_true = Tensor::<B, 1>::zeros([core_dim], device)
            .slice_assign([0..1], Tensor::<B, 1>::ones([1], device));
        let v_false = Tensor::<B, 1>::zeros([core_dim], device)
            .slice_assign([1..2], Tensor::<B, 1>::ones([1], device));
        Self {
            text_proc,
            vision_encoder,
            generator,
            v_true,
            v_false,
        }
    }

    pub fn forward(
        &self,
        texts: &[String],
        images: Tensor<B, 4>,
        memory: &GraphMemory<B>,
    ) -> Tensor<B, 2> {
        let text_seq_vecs = self.text_proc.forward(texts);
        let batch_size = texts.len();
        let core_dim = text_seq_vecs.dims()[2];

        let v_intent = text_seq_vecs
            .slice([0..batch_size, 0..1, 0..core_dim])
            .reshape([batch_size, core_dim]);

        let v_img_raw = self.vision_encoder.forward(images);

        let img_sq = (v_img_raw.clone() * v_img_raw.clone()).sum_dim(1);
        let v_img_entity = v_img_raw / (img_sq.sqrt() + 1e-8);

        let v_op = self.generator.forward(v_intent, memory);

        let (b_true, b_false) = self.get_truth_constants(batch_size);

        Operator::execute(v_op, v_img_entity, b_true, b_false)
    }

    pub fn get_truth_constants(&self, batch_size: usize) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let b_true = self
            .v_true
            .clone()
            .unsqueeze_dim(0)
            .repeat_dim(0, batch_size);
        let b_false = self
            .v_false
            .clone()
            .unsqueeze_dim(0)
            .repeat_dim(0, batch_size);
        (b_true, b_false)
    }
}
