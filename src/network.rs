use std::io::Write;

use crate::activation::{
    deserialize_activation_function, serialize_activation_function, ActivationFn,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct BackpropagationPassResult {
    layer_gradients: Vec<Vec<f64>>,
    total_error: f64,
}

pub struct LayerProperties {
    num_neurons: usize,
    activation: Box<dyn ActivationFn>,
}

impl LayerProperties {
    pub fn new(num_neurons: usize, activation: impl ActivationFn + 'static) -> Self {
        Self {
            num_neurons,
            activation: Box::new(activation),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    #[serde(
        serialize_with = "serialize_activation_function",
        deserialize_with = "deserialize_activation_function"
    )]
    activation: Box<dyn ActivationFn>,
}

impl Layer {
    pub fn new(
        weights: Vec<Vec<f64>>,
        biases: Vec<f64>,
        activation: impl ActivationFn + 'static,
    ) -> Self {
        Self {
            weights,
            biases,
            activation: Box::new(activation),
        }
    }

    pub fn with_dimensions(
        num_neurons: usize,
        num_inputs: usize,
        activation: Box<dyn ActivationFn>,
    ) -> Self {
        Self {
            weights: vec![vec![0.0; num_inputs]; num_neurons],
            biases: vec![0.0; num_neurons],
            activation,
        }
    }

    pub fn len(&self) -> usize {
        self.biases.len()
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Network<const NUM_IN: usize, const NUM_OUT: usize> {
    layers: Vec<Layer>,
}

impl<const NUM_IN: usize, const NUM_OUT: usize> Network<NUM_IN, NUM_OUT> {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn random(
        hidden_layers: Vec<LayerProperties>,
        output_activation: impl ActivationFn + 'static,
    ) -> Self {
        let mut layers = Vec::new();
        let mut prev_num_neurons = NUM_IN;
        for layer in hidden_layers {
            let mut weights = Vec::new();
            for _ in 0..layer.num_neurons {
                let mut row = Vec::new();
                for _ in 0..prev_num_neurons {
                    row.push(rand::random::<f64>() / (prev_num_neurons as f64));
                }
                weights.push(row);
            }
            let mut biases = Vec::new();
            for _ in 0..layer.num_neurons {
                biases.push(rand::random::<f64>() / (layer.num_neurons as f64));
            }
            layers.push(Layer {
                weights,
                biases,
                activation: layer.activation,
            });
            prev_num_neurons = layer.num_neurons;
        }

        let mut weights = Vec::new();
        for _ in 0..NUM_OUT {
            let mut row = Vec::new();
            for _ in 0..prev_num_neurons {
                row.push(rand::random::<f64>() / (prev_num_neurons as f64));
            }
            weights.push(row);
        }
        let mut biases = Vec::new();
        for _ in 0..NUM_OUT {
            biases.push(rand::random::<f64>() / (NUM_OUT as f64));
        }

        layers.push(Layer {
            weights,
            biases,
            activation: Box::new(output_activation),
        });

        return Self { layers };
    }

    pub fn forward(&self, input: [f64; NUM_IN]) -> [f64; NUM_OUT] {
        let mut last_output = input.to_vec();
        for (k, layer) in self.layers[..self.layers.len() - 1].iter().enumerate() {
            let mut new_output = Vec::new();
            for i in 0..layer.weights.len() {
                let mut sum = layer.biases[i];
                for j in 0..last_output.len() {
                    sum += layer.weights[i][j] * last_output[j];
                }
                new_output.push(layer.activation.forward(sum));
            }
            last_output = new_output;
        }
        let layer = &self.layers[self.layers.len() - 1];
        let mut output = [0.0; NUM_OUT];
        for i in 0..layer.weights.len() {
            let mut sum = layer.biases[i];
            for j in 0..last_output.len() {
                sum += layer.weights[i][j] * last_output[j];
            }
            output[i] = layer.activation.forward(sum);
        }
        return output;
    }

    pub fn forward_with_intermediate_outputs(&self, input: [f64; NUM_IN]) -> Vec<Vec<f64>> {
        let mut intermediate_outputs = vec![input.to_vec()];
        for layer in &self.layers {
            let last_output = intermediate_outputs.last().unwrap();
            let mut new_output = Vec::new();
            for i in 0..layer.weights.len() {
                let mut sum = layer.biases[i];
                for j in 0..last_output.len() {
                    sum += layer.weights[i][j] * last_output[j];
                }
                new_output.push(layer.activation.forward(sum));
            }
            intermediate_outputs.push(new_output);
        }
        intermediate_outputs
    }

    pub fn backpropagate(
        &mut self,
        input: [f64; NUM_IN],
        target: [f64; NUM_OUT],
        learning_rate: f64,
    ) -> BackpropagationPassResult {
        let intermediate_outputs = self.forward_with_intermediate_outputs(input);

        let mut layer_gradients = vec![vec![]; self.layers.len()];
        let mut total_error = 0.0;
        for i in 0..NUM_OUT {
            let output = intermediate_outputs.last().unwrap()[i];
            let error_gradient = output - target[i];
            layer_gradients.last_mut().unwrap().push(error_gradient);
            total_error += (-error_gradient) * (-error_gradient) / 2.0;
        }

        let mut layer_deltas = vec![];

        for i in (0..self.layers.len()).rev() {
            let layer = &self.layers[i];

            let mut layer_delta = Layer::with_dimensions(
                layer.weights.len(),
                layer.weights[0].len(),
                layer.activation.clone_to_box(),
            );
            if i > 0 {
                layer_gradients[i - 1] = vec![0.0; layer.weights[0].len()];
            }

            for j in 0..layer.len() {
                let output = intermediate_outputs[i + 1][j];
                let neuron_output_gradient = layer_gradients[i][j];
                let activation_gradient = layer.activation.derivative(output);
                let neuron_net_input_gradient = neuron_output_gradient * activation_gradient;

                let mut neuron_weights_delta = Vec::new();

                for k in 0..layer.weights[j].len() {
                    let weight_gradient = neuron_net_input_gradient * intermediate_outputs[i][k];
                    let layer_delta = learning_rate * weight_gradient;
                    neuron_weights_delta.push(layer_delta);
                    if i > 0 {
                        layer_gradients[i - 1][k] += layer.weights[j][k] * neuron_net_input_gradient;
                    }
                }

                let bias_gradient = neuron_net_input_gradient;
                let bias_delta = learning_rate * bias_gradient;
                layer_delta.biases[j] = bias_delta;

                layer_delta.weights[j] = neuron_weights_delta;
            }
            layer_deltas.push(layer_delta);
        }

        for i in 0..self.layers.len() {
            let layer_count = self.layers.len();
            let layer = &mut self.layers[i];
            for j in 0..layer.len() {
                for k in 0..layer.weights[j].len() {
                    layer.weights[j][k] -= layer_deltas[layer_count - i - 1].weights[j][k];
                }
                layer.biases[j] -= layer_deltas[layer_count - i - 1].biases[j];
            }
        }

        return BackpropagationPassResult {
            layer_gradients,
            total_error,
        };
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let serialized = bincode::serialize(self).or_else(|e| {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(&serialized)?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let deserialized: Self = bincode::deserialize_from(file).or_else(|e| {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;
        Ok(deserialized)
    }

    pub fn save_json_to_file(&self, path: &str) -> std::io::Result<()> {
        let serialized = serde_json::to_string(self).or_else(|e| {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(&serialized.as_bytes())?;
        Ok(())
    }

    pub fn load_json_from_file(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let deserialized: Self = serde_json::from_reader(file).or_else(|e| {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;
        Ok(deserialized)
    }
}
