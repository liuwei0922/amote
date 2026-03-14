use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::Int;

use plotters::prelude::*;
use rand::prelude::*;
use std::collections::HashMap;

use amote::processor::{
    CoreProcessorConfig, GraphMemory, MatchOutputProcessorConfig, RouterConfig,
    StateInputProcessorConfig, System, TextInputProcessor, TextTranslatorConfig,
};

const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 4000;
const LEARNING_RATE: f64 = 0.001;
const CORE_DIM: usize = 128;
const EMBED_DIM: usize = 200;

fn get_world_rules() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("左", "WEST"),
        ("右", "EAST"),
        ("前", "NORTH"),
        ("后", "SOUTH"),
        ("向左", "WEST"),
        ("向右", "EAST"),
    ])
}

const ALL_STATES: [&str; 4] = ["NORTH", "SOUTH", "EAST", "WEST"];

type TrainBackend = Autodiff<NdArray<f32>>;

#[derive(Module, Debug)]
pub struct AgentWeights<B: Backend> {
    pub text_model: amote::processor::TextTranslator<B>,
    pub state_proc: amote::processor::StateInputProcessor<B>,
    pub core: amote::processor::CoreProcessor<B>,
    pub router: amote::processor::Router<B>,
    pub outputs: Vec<amote::processor::MatchOutputProcessor<B>>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== AGI System: 端到端动态认知网络训练 (纯净版) ===");

    let device = Default::default();

    let rules = get_world_rules();
    let all_insts: Vec<&str> = rules.keys().cloned().collect();

    let mut vocab = HashMap::new();
    for (i, &w) in all_insts.iter().enumerate() {
        vocab.insert(w.to_string(), i);
    }
    let unk_index = vocab.len();

    let static_embeddings = Tensor::<TrainBackend, 2>::random(
        [unk_index + 1, EMBED_DIM],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let text_proc = TextInputProcessor::from_parts(
        vocab,
        static_embeddings,
        unk_index,
        TextTranslatorConfig::new(EMBED_DIM, CORE_DIM).init(&device),
        device.clone(),
    );

    let state_proc = StateInputProcessorConfig::new(CORE_DIM).init(&device);
    let outputs = vec![MatchOutputProcessorConfig::new(CORE_DIM).init(&device)];
    let core = CoreProcessorConfig::new(CORE_DIM).init(&device);
    let router = RouterConfig::new(CORE_DIM, 2, outputs.len()).init(&device);
    let memory = GraphMemory::<TrainBackend>::new(CORE_DIM, &device);

    let mut system = System::from_parts(text_proc, state_proc, outputs, core, router, memory);

    // let mut system = System::<TrainBackend>::new(
    //     "./word/tencent_vocab.json",
    //     "./word/tencent_w2v.safetensors",
    //     &device,
    // )?;

    let mut optim = AdamWConfig::new().with_weight_decay(0.0).init();
    let ce_loss = CrossEntropyLossConfig::new().init(&device);

    let mut history_loss = Vec::new();
    let mut history_acc = Vec::new();
    let mut rng = rand::rng();

    for epoch in 0..EPOCHS {
        let mut batch_insts = Vec::with_capacity(BATCH_SIZE);
        let mut batch_states = Vec::with_capacity(BATCH_SIZE);
        let mut batch_targets: Vec<i64> = Vec::with_capacity(BATCH_SIZE);

        for _ in 0..BATCH_SIZE {
            if rng.random_bool(0.5) {
                let inst = all_insts.choose(&mut rng).unwrap();
                let state = rules.get(inst).unwrap();
                batch_insts.push(inst.to_string());
                batch_states.push(state.to_string());
                batch_targets.push(0i64);
            } else {
                let inst = all_insts.choose(&mut rng).unwrap();
                let mut state = ALL_STATES.choose(&mut rng).unwrap();
                let correct_state = rules.get(inst).unwrap();
                while state == correct_state {
                    state = ALL_STATES.choose(&mut rng).unwrap();
                }
                batch_insts.push(inst.to_string());
                batch_states.push(state.to_string());
                batch_targets.push(1i64);
            }
        }

        let targets = Tensor::<TrainBackend, 1, Int>::from_ints(batch_targets.as_slice(), &device);

        let (results, _) = system.forward(&batch_insts, &batch_states, true);

        let logits_3d = results[0].clone().expect("训练模式下应该有输出");
        let logits = logits_3d.reshape([BATCH_SIZE, 2]);

        let loss = ce_loss.forward(logits.clone(), targets.clone());
        let grads = loss.backward();

        let weights = AgentWeights {
            text_model: system.text_proc.model.clone(),
            state_proc: system.state_proc.clone(),
            core: system.core.clone(),
            router: system.router.clone(),
            outputs: system.outputs.clone(),
        };

        let grads_params = GradientsParams::from_grads(grads, &weights);
        let updated_weights = optim.step(LEARNING_RATE, weights, grads_params);

        system.text_proc.model = updated_weights.text_model;
        system.state_proc = updated_weights.state_proc;
        system.core = updated_weights.core;
        system.router = updated_weights.router;
        system.outputs = updated_weights.outputs;

        let preds = logits.argmax(1);
        let preds_vec = preds.into_data().to_vec::<i64>().unwrap();

        let mut correct_mask = vec![false; BATCH_SIZE];
        let mut correct_count = 0;
        for i in 0..BATCH_SIZE {
            if preds_vec[i] == batch_targets[i] {
                correct_mask[i] = true;
                correct_count += 1;
            }
        }
        let acc = correct_count as f32 / BATCH_SIZE as f32;

        if epoch > 100 {
            system.consolidate_memory(&correct_mask);
        }

        let loss_val = loss.into_data().to_vec::<f32>().unwrap()[0];
        history_loss.push(loss_val);
        history_acc.push(acc);

        if (epoch + 1) % 50 == 0 {
            println!(
                "Epoch {:4} | Batch Loss: {:.4} | Batch Acc: {:.2}",
                epoch + 1,
                loss_val,
                acc
            );
        }
    }

    draw_training_curve(&history_loss, &history_acc)?;

    println!("\n=== 泛化测试 (Zero-Shot) ===");
    let test_cases = vec![
        ("左", "WEST", true),
        ("右", "NORTH", false),
        ("前", "NORTH", true),
        ("后", "SOUTH", true),
        ("向左", "WEST", true),
        ("向右", "EAST", true),
    ];

    for (t, s, truth) in test_cases {
        let inputs_t = vec![t.to_string()];
        let inputs_s = vec![s.to_string()];

        let (results, _) = system.forward(&inputs_t, &inputs_s, false);

        if let Some(logits_3d) = &results[0] {
            let pred_vec = logits_3d
                .clone()
                .reshape([1, 2])
                .argmax(1)
                .into_data()
                .to_vec::<i64>()
                .unwrap();
            let pred_idx = pred_vec[0];

            let ans = if pred_idx == 0 { "MATCH" } else { "MISMATCH" };
            let expect = if truth { "MATCH" } else { "MISMATCH" };

            println!(
                "指令:'{}', 状态:'{}' => AI判断: {} (预期: {})",
                t, s, ans, expect
            );
        } else {
            println!("指令:'{}' => 路由错误，未选中任何输出模块", t);
        }
    }

    Ok(())
}

fn draw_training_curve(loss: &[f32], acc: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n>>> 正在生成训练曲线图 (training_curve.png)...");
    let root = BitMapBackend::new("training_curve.png", (1024, 500)).into_drawing_area();
    root.fill(&WHITE)?;
    let (left, right) = root.split_horizontally(512);

    let mut chart_loss = ChartBuilder::on(&left)
        .caption("Training Loss", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..loss.len() as f32, 0f32..1.5f32)?;
    chart_loss.configure_mesh().draw()?;
    chart_loss.draw_series(LineSeries::new(
        loss.iter().enumerate().map(|(i, &v)| (i as f32, v)),
        &BLUE,
    ))?;

    let mut chart_acc = ChartBuilder::on(&right)
        .caption("Training Accuracy", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..acc.len() as f32, 0f32..1.1f32)?;
    chart_acc.configure_mesh().draw()?;
    chart_acc.draw_series(LineSeries::new(
        acc.iter().enumerate().map(|(i, &v)| (i as f32, v)),
        &GREEN,
    ))?;

    println!("✅ 图片已保存！");
    Ok(())
}
