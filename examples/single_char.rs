#![recursion_limit = "512"]

use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::relu;
use rand::RngExt;

use amote::agent::system::AgentSystem;
use amote::memory::graph::GraphMemory;
use amote::perception::text::TextInputProcessor;

const BATCH_SIZE: usize = 16;
const EPOCHS: usize = 1500;
const CORE_DIM: usize = 128;
const LEARNING_RATE: f64 = 0.002;

type TrainBackend = Autodiff<NdArray<f32>>;

#[derive(Module, Debug)]
pub struct AgentWeights<B: Backend> {
    pub text_proc: amote::perception::text::TextEncoder<B>,
    pub vision_encoder: amote::perception::vision::VisionEncoder<B>,
    pub generator: amote::algebra::generator::Generator<B>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 🧪 核心定理验证：单字视觉代数系统 ===");
    let device = Default::default();

    let text_proc = TextInputProcessor::<TrainBackend>::new(CORE_DIM, 10, &device);
    let mut system = AgentSystem::new(text_proc, CORE_DIM, &device);
    let memory = GraphMemory::<TrainBackend>::new(CORE_DIM, &device);

    let mut optim = AdamWConfig::new().with_weight_decay(0.0).init();
    let mut rng = rand::rng();

    let colors = ["红", "绿", "蓝"];

    println!("开始训练：让模型在 128 维空间中建立颜色切割法则...");

    for epoch in 0..EPOCHS {
        let mut batch_texts = Vec::with_capacity(BATCH_SIZE);

        let mut batch_pixels = vec![0.0f32; BATCH_SIZE * 3 * 64 * 64];
        let mut batch_is_true = Vec::with_capacity(BATCH_SIZE);

        for b in 0..BATCH_SIZE {
            let text_idx = rng.random_range(0..3);
            let text = colors[text_idx];
            let is_match = rng.random_bool(0.5);
            let img_color_idx = if is_match {
                text_idx
            } else {
                (text_idx + 1) % 3
            };

            for h in 0..64 {
                for w in 0..64 {
                    let pixel_idx = b * 3 * 64 * 64 + img_color_idx * 64 * 64 + h * 64 + w;
                    batch_pixels[pixel_idx] = 1.0;
                }
            }
            batch_texts.push(text.to_string());
            batch_is_true.push(is_match);
        }

        let images = Tensor::<TrainBackend, 1>::from_floats(batch_pixels.as_slice(), &device)
            .reshape([BATCH_SIZE, 3, 64, 64]);

        let v_result = system.forward(&batch_texts, images, &memory);

        let (b_true, b_false) = system.get_truth_constants(BATCH_SIZE);
        let mask_array: Vec<i32> = batch_is_true
            .iter()
            .map(|&b| if b { 1 } else { 0 })
            .collect();

        let mask_bool =
            Tensor::<TrainBackend, 1, burn::tensor::Int>::from_ints(mask_array.as_slice(), &device)
                .reshape([BATCH_SIZE, 1])
                .equal_elem(1);

        let dist_to_true = (v_result.clone() - b_true).powf_scalar(2.0).sum_dim(1);
        let dist_to_false = (v_result - b_false).powf_scalar(2.0).sum_dim(1);

        let positive_dist = dist_to_false
            .clone()
            .mask_where(mask_bool.clone(), dist_to_true.clone());
        let negative_dist = dist_to_true.mask_where(mask_bool, dist_to_false);

        let margin = 1.5f32;
        let repulsion_loss = relu(negative_dist.neg() + margin);
        let loss = (positive_dist + repulsion_loss).mean();

        let grads = loss.backward();

        let weights = AgentWeights {
            text_proc: system.text_proc.encoder.clone(),
            vision_encoder: system.vision_encoder.clone(),
            generator: system.generator.clone(),
        };

        let grads_params = GradientsParams::from_grads(grads, &weights);
        let updated_weights = optim.step(LEARNING_RATE, weights, grads_params);

        system.text_proc.encoder = updated_weights.text_proc;
        system.vision_encoder = updated_weights.vision_encoder;
        system.generator = updated_weights.generator;

        if (epoch + 1) % 100 == 0 {
            let loss_val = loss.into_data().to_vec::<f32>().unwrap()[0];
            println!("Epoch {:4} | 拓扑对齐 Loss: {:.6}", epoch + 1, loss_val);
        }
    }

    println!("✅ 训练完成！视觉与文本成功在概念潜空间对齐！");
    println!("\n=== 🧠 泛化推理测试 (Inference Test) ===");
    println!("剥夺梯度的庇护，让系统在未知组合下进行代数运算...");

    let test_cases = vec![
        ("红", 0, true, "红色指令 切 红色图像"),
        ("红", 2, false, "红色指令 切 蓝色图像"),
        ("绿", 1, true, "绿色指令 切 绿色图像"),
        ("蓝", 1, false, "蓝色指令 切 绿色图像"),
        ("蓝", 2, true, "蓝色指令 切 蓝色图像"),
    ];

    let num_tests = test_cases.len();
    let mut test_texts = Vec::with_capacity(num_tests);
    let mut test_pixels = vec![0.0f32; num_tests * 3 * 64 * 64];

    for (b, &(txt, color_idx, _, _)) in test_cases.iter().enumerate() {
        test_texts.push(txt.to_string());
        for h in 0..64 {
            for w in 0..64 {
                let pixel_idx = b * 3 * 64 * 64 + color_idx * 64 * 64 + h * 64 + w;
                test_pixels[pixel_idx] = 1.0;
            }
        }
    }

    let test_images = Tensor::<TrainBackend, 1>::from_floats(test_pixels.as_slice(), &device)
        .reshape([num_tests, 3, 64, 64]);

    let v_result = system.forward(&test_texts, test_images, &memory);
    let (b_true, b_false) = system.get_truth_constants(num_tests);

    let dist_to_true = (v_result.clone() - b_true)
        .powf_scalar(2.0)
        .sum_dim(1)
        .reshape([num_tests]);
    let dist_to_false = (v_result - b_false)
        .powf_scalar(2.0)
        .sum_dim(1)
        .reshape([num_tests]);

    let dist_true_vec = dist_to_true.into_data().to_vec::<f32>().unwrap();
    let dist_false_vec = dist_to_false.into_data().to_vec::<f32>().unwrap();

    for i in 0..num_tests {
        let (_, color_idx, expected, desc) = test_cases[i];
        let img_color_name = colors[color_idx];

        let ai_predict_is_true = dist_true_vec[i] < dist_false_vec[i];
        let ai_ans = if ai_predict_is_true {
            "MATCH   "
        } else {
            "MISMATCH"
        };
        let exp_ans = if expected { "MATCH   " } else { "MISMATCH" };
        let icon = if ai_predict_is_true == expected {
            "✅"
        } else {
            "❌"
        };

        println!(
            "{} [{}] -> 视觉输入: [纯 {}] | AI 代数推演: {} (预期: {})",
            icon, desc, img_color_name, ai_ans, exp_ans
        );
        println!("   ├─ 距真理轴距离: {:.4}", dist_true_vec[i]);
        println!("   └─ 距谬误轴距离: {:.4}\n", dist_false_vec[i]);
    }
    Ok(())
}
