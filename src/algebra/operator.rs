use burn::prelude::*;
use burn::tensor::activation::sigmoid;

pub struct Operator;

impl Operator {
    pub fn execute<B: Backend>(
        v_op: Tensor<B, 2>,
        v_target: Tensor<B, 2>,
        v_true: Tensor<B, 2>,
        v_false: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let dot_product = (v_op.clone() * v_target.clone()).sum_dim(1);

        let threshold = 0.5;
        let sharpness = 20.0;

        let boolean_score = sigmoid((dot_product - threshold) * sharpness);

        let score_true = boolean_score.clone();
        let score_false = boolean_score.neg() + 1.0;

        let v_result = (v_true * score_true) + (v_false * score_false);

        v_result
    }
}
