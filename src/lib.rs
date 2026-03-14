pub mod processor;

pub mod perception;
pub mod algebra;
pub mod memory;
pub mod agent;

// 预导出最常用的核心组件，方便外部 (如 examples) 直接调用
// pub use perception::text::{TextEncoder, TextEncoderConfig};
// pub use algebra::generator::{Generator, GeneratorConfig};
// pub use algebra::operator::Operator;
// pub use memory::graph::GraphMemory;
// pub use agent::system::System;