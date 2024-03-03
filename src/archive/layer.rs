use std::usize;

use crate::activation::ActivationFn;

pub trait LayerTrait<const NUM_IN: usize> {
    fn forward(&self, input: &[f64]) -> Vec<f64>;


    fn clone_to_box(self: &Self) -> Box<dyn LayerTrait<NUM_IN>>;
}


pub struct Layer<const NUM_IN: usize, const NUM_OUT: usize> {
    weights: [[f64; NUM_IN]; NUM_OUT],
    biases: [f64; NUM_OUT],
    activation: Box<dyn ActivationFn>,
    next: Option<Box<dyn LayerTrait<NUM_OUT>>>,
}

impl<const NUM_IN: usize, const NUM_OUT: usize> Layer<NUM_IN, NUM_OUT> {
    // pub fn new(
    //     weights: [[f64; NUM_IN]; NUM_OUT],
    //     biases: [f64; NUM_OUT],
    //     activation: Box<dyn ActivationFn>,
    //     next: Option<Box<dyn LayerTrait<NUM_OUT>>>,
    // ) -> Self {
    //     Self {
    //         weights,
    //         biases,
    //         activation,
    //         next,
    //     }
    // }

    pub fn random(activation: impl ActivationFn + 'static) -> Self {
        let mut weights = [[0.0; NUM_IN]; NUM_OUT];
        let mut biases = [0.0; NUM_OUT];
        for i in 0..NUM_OUT {
            biases[i] = rand::random();
            for j in 0..NUM_IN {
                weights[i][j] = rand::random::<f64>() / (NUM_IN as f64);
            }
        }
        Self {
            weights,
            biases,
            activation: Box::new(activation),
            next: None,
        }
    }

    pub fn with_next<const NUM_NEXT_OUT: usize>(self, next: Layer<NUM_OUT, NUM_NEXT_OUT>) -> Self {
        Self {
            next: Some(next.clone_to_box()),
            ..self
        }
    }

    pub fn run(&self, input: [f64; NUM_IN]) -> Vec<f64> {
        self.forward(&input)
    }
}

impl<const NUM_IN: usize, const NUM_OUT: usize> Clone for Layer<NUM_IN, NUM_OUT> {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights,
            biases: self.biases,
            activation: self.activation.clone_to_box(),
            next: self.next.as_ref().and_then(|x| Some(x.clone_to_box())),
        }
    }
}

impl<const NUM_IN: usize, const NUM_OUT: usize> LayerTrait<NUM_IN> for Layer<NUM_IN, NUM_OUT> {
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; NUM_OUT];
        for i in 0..NUM_OUT {
            output[i] = self.biases[i];
            for j in 0..NUM_IN {
                output[i] += self.weights[i][j] * input[j];
            }
            output[i] = self.activation.forward(output[i]);
        }
        if let Some(next) = &self.next {
            return next.forward(&output);
        }
        output
    }

    fn clone_to_box(self: &Self) -> Box<dyn LayerTrait<NUM_IN>> {
        Box::new(Self {
            weights: self.weights.clone(),
            biases: self.biases.clone(),
            activation: self.activation.clone_to_box(),
            next: self.next.as_ref().and_then(|x| Some(x.clone_to_box())),
        })
    }
}
