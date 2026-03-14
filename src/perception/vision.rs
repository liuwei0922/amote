use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::gelu;

#[derive(Config, Debug)]
pub struct VisionEncoderConfig {
    pub core_dim: usize,
    #[config(default = 64)]
    pub image_size: usize,
}

#[derive(Module, Debug)]
pub struct VisionEncoder<B: Backend> {
    conv1: Conv2d<B>,
    pool1: MaxPool2d,
    conv2: Conv2d<B>,
    pool2: MaxPool2d,
    fc: Linear<B>,
}

impl VisionEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VisionEncoder<B> {
        let conv1 = Conv2dConfig::new([3, 16], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Valid)
            .init(device);
        let pool1 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let conv2 = Conv2dConfig::new([16, 32], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Valid)
            .init(device);
        let pool2 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let flat_size = 32 * 14 * 14;

        let fc = LinearConfig::new(flat_size, self.core_dim).init(device);

        VisionEncoder {
            conv1,
            pool1,
            conv2,
            pool2,
            fc,
        }
    }
}

impl<B: Backend> VisionEncoder<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(images);
        let x = gelu(x);
        let x = self.pool1.forward(x);

        let x = self.conv2.forward(x);
        let x = gelu(x);
        let x = self.pool2.forward(x);

        let x = x.flatten(1, 3);

        self.fc.forward(x)
    }
}
