use std::fmt::Debug;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Serialize, Deserialize, Debug)]
pub enum ActivationFnInfo {
    Identity,
    Sigmoid,
    DebugPrint {
        output: String,
        activation: Box<ActivationFnInfo>,
    },
}

impl ActivationFnInfo {
    pub fn get_function(&self) -> Box<dyn ActivationFn> {
        match self {
            ActivationFnInfo::Identity => Box::new(Identity),
            ActivationFnInfo::Sigmoid => Box::new(Sigmoid),
            ActivationFnInfo::DebugPrint { output, activation } => Box::new(DebugPrint {
                output: output.clone(),
                activation: activation.get_function(),
            }),
        }
    }
}

pub fn serialize_activation_function<S>(
    activation: &Box<dyn ActivationFn>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let activation_info = activation.get_serializable();
    ActivationFnInfo::serialize(&activation_info, serializer)
}

pub fn deserialize_activation_function<'de, D>(
    deserializer: D,
) -> Result<Box<dyn ActivationFn>, D::Error>
where
    D: Deserializer<'de>,
{
    let activation_info = ActivationFnInfo::deserialize(deserializer)?;
    Ok(activation_info.get_function())
}

pub trait ActivationFn {
    fn forward(&self, x: f64) -> f64;

    fn backward(&self, x: f64) -> f64;

    fn derivative(&self, x: f64) -> f64;

    fn clone_to_box(&self) -> Box<dyn ActivationFn>;

    fn get_serializable(&self) -> ActivationFnInfo;
}

impl Debug for dyn ActivationFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.get_serializable().fmt(f)
    }
}

pub struct Identity;

impl ActivationFn for Identity {
    fn forward(&self, x: f64) -> f64 {
        x
    }

    fn backward(&self, x: f64) -> f64 {
        x
    }

    fn derivative(&self, _x: f64) -> f64 {
        1.0
    }

    fn clone_to_box(&self) -> Box<dyn ActivationFn> {
        Box::new(Identity)
    }

    fn get_serializable(&self) -> ActivationFnInfo {
        ActivationFnInfo::Identity
    }
}

pub struct Sigmoid;

impl ActivationFn for Sigmoid {
    fn forward(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn backward(&self, x: f64) -> f64 {
        -((1.0 - x) / x).ln()
    }

    fn derivative(&self, x: f64) -> f64 {
        x * (1.0 - x)
    }

    fn clone_to_box(&self) -> Box<dyn ActivationFn> {
        Box::new(Sigmoid)
    }

    fn get_serializable(&self) -> ActivationFnInfo {
        ActivationFnInfo::Sigmoid
    }
}

pub struct DebugPrint {
    output: String,
    activation: Box<dyn ActivationFn>,
}

#[allow(dead_code)]
impl DebugPrint {
    pub fn new(output: String, activation: impl ActivationFn + 'static) -> Self {
        Self {
            output,
            activation: Box::new(activation),
        }
    }
}

impl ActivationFn for DebugPrint {
    fn forward(&self, x: f64) -> f64 {
        let result = self.activation.forward(x);
        println!("{}; f({}) = {}", self.output, x, result);
        result
    }

    fn backward(&self, x: f64) -> f64 {
        let result = self.activation.backward(x);
        println!("{}; f⁻¹({}) = {}", self.output, x, result);
        result
    }

    fn derivative(&self, x: f64) -> f64 {
        let result = self.activation.derivative(x);
        println!("{}; f'({}) = {}", self.output, x, result);
        result
    }

    fn clone_to_box(&self) -> Box<dyn ActivationFn> {
        Box::new(DebugPrint {
            output: self.output.clone(),
            activation: self.activation.clone_to_box(),
        })
    }

    fn get_serializable(&self) -> ActivationFnInfo {
        ActivationFnInfo::DebugPrint {
            output: self.output.clone(),
            activation: Box::new(self.activation.get_serializable()),
        }
    }
}
