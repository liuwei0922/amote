use burn::prelude::*;

pub struct Operator;

impl Operator {
    pub fn execute<B: Backend>(v_op: Tensor<B, 2>, v_target: Tensor<B, 2>) -> Tensor<B, 2> {
        (v_op * v_target).sum_dim(1)
    }
}
