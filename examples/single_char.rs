#![recursion_limit = "512"] 

use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use rand::RngExt;

use amote::agent::system::AgentSystem;
use amote::memory::graph::GraphMemory;
use amote::perception::text::TextInputProcessor;

const BATCH_SIZE: usize = 16;
const EPOCHS: usize = 1500;
const CORE_DIM: usize = 128;
const LEARNING_RATE: f64 = 0.005;

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
    
    let mut optim = AdamWConfig::new().with_weight_decay(1e-4).init();
    let mut rng = rand::rng();
    let colors = ["红", "绿", "蓝"];
    
    println!("开始训练：建立正交基底 (1.0 代表命中，0.0 代表正交排斥)...");
    
    for epoch in 0..EPOCHS {
        let mut batch_texts = Vec::with_capacity(BATCH_SIZE);
        let mut batch_pixels = vec![0.0f32; BATCH_SIZE * 3 * 64 * 64]; 
        let mut batch_is_true = Vec::with_capacity(BATCH_SIZE);
        
        for b in 0..BATCH_SIZE {
            let text_idx = rng.random_range(0..3);
            let text = colors[text_idx];
            let is_match = rng.random_bool(0.5);
            let img_color_idx = if is_match { text_idx } else { (text_idx + 1) % 3 };
            
            for h in 0..64 {
                for w in 0..64 {
                    let pixel_idx = b * 3 * 64 * 64 + img_color_idx * 64 * 64 + h * 64 + w;
                    batch_pixels[pixel_idx] = 1.0;
                }
            }
            batch_texts.push(text.to_string());
            batch_is_true.push(if is_match { 1.0f32 } else { 0.0f32 });
        }
        
        let images = Tensor::<TrainBackend, 1>::from_floats(batch_pixels.as_slice(), &device)
            .reshape([BATCH_SIZE, 3, 64, 64]);
            
        let v_result = system.forward(&batch_texts, images, &memory);
        
        let target = Tensor::<TrainBackend, 1>::from_floats(batch_is_true.as_slice(), &device)
            .reshape([BATCH_SIZE, 1]);

        let loss = (v_result - target).powf_scalar(2.0).mean();
        
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
            println!("Epoch {:4} | 正交对齐 Loss: {:.6}", epoch + 1, loss_val);
        }
    }
    
    println!("✅ 训练完成！底层判断逻辑已建立！");

    println!("\n=== 🌌 高阶代数提取测试 (What Color is it?) ===");
    
    let base_texts = vec!["红".to_string(), "绿".to_string(), "蓝".to_string()];
    let base_seqs = system.text_proc.forward(&base_texts);
    let v_intent_base = base_seqs.slice([0..3, 0..1, 0..CORE_DIM]).reshape([3, CORE_DIM]);
    let v_base = system.generator.forward(v_intent_base, &memory); 
    
    let v_red   = v_base.clone().slice([0..1]).reshape([CORE_DIM, 1]);
    let v_green = v_base.clone().slice([1..2]).reshape([CORE_DIM, 1]);
    let v_blue  = v_base.clone().slice([2..3]).reshape([CORE_DIM, 1]);

    let m_red   = v_red.clone().matmul(v_red.clone().transpose());
    let m_green = v_green.clone().matmul(v_green.clone().transpose());
    let m_blue  = v_blue.clone().matmul(v_blue.clone().transpose());
    let m_color = m_red + m_green + m_blue; 

    let what_tests = vec![("色", 0, "红"), ("色", 1, "绿"), ("色", 2, "蓝")];
    let num_what = what_tests.len();
    let mut test_pixels_what = vec![0.0f32; num_what * 3 * 64 * 64];

    for (b, &(_, color_idx, _)) in what_tests.iter().enumerate() {
        for h in 0..64 {
            for w in 0..64 {
                let pixel_idx = b * 3 * 64 * 64 + color_idx * 64 * 64 + h * 64 + w;
                test_pixels_what[pixel_idx] = 1.0;
            }
        }
    }

    let images_what = Tensor::<TrainBackend, 1>::from_floats(test_pixels_what.as_slice(), &device)
        .reshape([num_what, 3, 64, 64]);
    
    let v_img_raw = system.vision_encoder.forward(images_what);
    let img_sq = (v_img_raw.clone() * v_img_raw.clone()).sum_dim(1);
    let v_img_entity = v_img_raw / (img_sq.sqrt() + 1e-8); 

    let v_extracted = v_img_entity.matmul(m_color); 

    for i in 0..num_what {
        let single_extracted = v_extracted.clone().slice([i..i+1]); 
        let mut min_dist = f32::MAX;
        let mut predicted_color = "";

        for j in 0..3 {
            let single_v_base = v_base.clone().slice([j..j+1]); 
            let dist = (single_extracted.clone() - single_v_base).powf_scalar(2.0).sum_dim(1).into_scalar().to_f32();
            if dist < min_dist {
                min_dist = dist;
                predicted_color = colors[j];
            }
        }
        let expected_color = what_tests[i].2;
        let icon = if predicted_color == expected_color { "✅" } else { "❌" };
        println!("{} 视觉输入: [{}色图像] -> M_色(图) 提取结果匹配到: [V_{}]", icon, expected_color, predicted_color);
    }
    Ok(())
}