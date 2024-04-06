use crate::value::Value;

struct Layer {
    weights: Vec<Vec<Value>>,
    biases: Vec<Value>,
    activation: fn(Value) -> Value,
}

pub struct NN {
    layers: Vec<Layer>,
}

impl NN {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, shape: (usize, usize), activation: fn(Value) -> Value) {
        let (input_size, output_size) = shape;
        if let Some(prev_layer) = self.layers.last() {
            assert_eq!(prev_layer.weights.len(), input_size);
        }
        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| Value::new(rand::random::<f32>() - 0.5, None))
                    .collect()
            })
            .collect();
        let biases = (0..output_size)
            .map(|_| Value::new(rand::random::<f32>() - 0.5, None))
            .collect();
        self.layers.push(Layer {
            weights,
            biases,
            activation,
        });
    }

    pub fn forward(&self, input: Vec<Value>) -> Vec<Value> {
        let mut output = input;
        for layer in &self.layers {
            output = layer
                .weights
                .iter()
                .map(|weights| {
                    weights
                        .iter()
                        .zip(&output)
                        .map(|(w, i)| w * i)
                        .sum::<Value>()
                })
                .zip(&layer.biases)
                .map(|(value, bias)| (layer.activation)(&value + &bias))
                .collect();
        }
        output
    }

    pub fn backward(&self, output: &mut Vec<Value>) {
        output.iter_mut().for_each(|output| output.backward());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nn() {
        let mut nn = NN::new();
        nn.add_layer((2, 3), |x| x);
        nn.add_layer((3, 1), |x| x);
        let input = vec![Value::new(1.0, None), Value::new(2.0, None)];
        let output = nn.forward(input);
        assert_eq!(output.len(), 1);
    }
}
