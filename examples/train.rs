use anyhow::Result;
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use plotters::prelude::*;
use rand::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use amote::processor::System;

const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 10000;
const LEARNING_RATE: f64 = 0.005;

fn get_world_rules() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("左", "WEST"),
        ("右", "EAST"),
        ("前", "NORTH"),
        ("后", "SOUTH"),
    ])
}

const ALL_STATES: [&str; 4] = ["NORTH", "SOUTH", "EAST", "WEST"];

fn main() -> Result<()> {
    println!("=== AGI System: 端到端动态算子网络训练 (Rust版) ===");

    let device = Device::Cpu;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let system = System::new(
        vs,
        "./word/tencent_vocab.json",
        "./word/tencent_w2v.safetensors",
    )?;

    let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), LEARNING_RATE)?;

    let mut history_loss = Vec::new();
    let mut history_acc = Vec::new();

    let rules = get_world_rules();
    let all_insts: Vec<&str> = rules.keys().cloned().collect();
    let mut rng = rand::rng();

    for epoch in 0..EPOCHS {
        let mut batch_insts = Vec::with_capacity(BATCH_SIZE);
        let mut batch_states = Vec::with_capacity(BATCH_SIZE);
        let mut batch_targets = Vec::with_capacity(BATCH_SIZE);

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

        let targets = Tensor::from_slice(&batch_targets, (BATCH_SIZE,), &device)?;

        let (results, _) = system.forward(&batch_insts, &batch_states, true)?;

        let logits = results[0].as_ref().unwrap();
        let logits = logits.squeeze(1)?;

        let loss = candle_nn::loss::cross_entropy(&logits, &targets)?;

        optimizer.backward_step(&loss)?;

        let preds = logits.argmax(D::Minus1)?;
        let correct_mask = preds.eq(&targets.to_dtype(DType::U32)?)?;
        let acc = correct_mask
            .to_dtype(DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()?;

        system.consolidate_memory(Some(&correct_mask))?;

        history_loss.push(loss.to_scalar::<f32>()?);
        history_acc.push(acc);

        if (epoch + 1) % 200 == 0 {
            println!(
                "Epoch {:4} | Batch Loss: {:.4} | Batch Acc: {:.2}",
                epoch + 1,
                loss.to_scalar::<f32>()?,
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

        let (results, _) = system.forward(&inputs_t, &inputs_s, false)?;

        if let Some(logits) = &results[0] {
            let pred_idx = logits.squeeze(1)?.argmax(D::Minus1)?.to_scalar::<u32>()?;
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

fn draw_training_curve(loss: &[f32], acc: &[f32]) -> Result<()> {
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

fn ensure_dummy_resources() -> Result<()> {
    let vocab_path = "train_vocab.json";
    let model_path = "train_model.safetensors";

    if Path::new(vocab_path).exists() && Path::new(model_path).exists() {
        return Ok(());
    }

    println!("生成临时 Vocab 和 Safetensors...");

    let words = vec!["左", "右", "前", "后", "向", "向左", "向右"];
    let mut vocab = HashMap::new();
    for (i, w) in words.iter().enumerate() {
        vocab.insert(w.to_string(), i);
    }

    let f = File::create(vocab_path)?;
    serde_json::to_writer(f, &vocab)?;

    let device = Device::Cpu;
    let weight = Tensor::randn(0f32, 1f32, (words.len(), 200), &device)?;

    let map = HashMap::from([("weight".to_string(), weight)]);
    candle_core::safetensors::save(&map, model_path)?;

    Ok(())
}
