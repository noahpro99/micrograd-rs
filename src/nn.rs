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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nn() {
        let mut nn = NN::new();
        nn.add_layer((2, 3), Value::relu);
        nn.add_layer((3, 1), Value::relu);
        let input = vec![Value::new(1.0, None), Value::new(2.0, None)];
        let output = nn.forward(input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn small_dataset() {
        fn test_fn((x, y): (&Value, &Value)) -> Value {
            let raw = &(&x.pow(2) + &y.pow(2)) - &Value::new(3.0, None);
            Value::new(
                if raw.value.borrow().is_sign_positive() {
                    1.0
                } else {
                    -1.0
                },
                None,
            )
        }

        // points around -3 to 3 in x and y
        let grid = (-3..=3)
            .flat_map(|x| {
                (-3..=3).map(move |y| (Value::new(x as f32, None), Value::new(y as f32, None)))
            })
            .collect::<Vec<_>>();
        let test_data = grid.iter().map(|(x, y)| (x, y, test_fn((x, y))));

        dbg!(test_data
            .map(|(x, y, z)| {
                (
                    x.value.borrow().to_owned(),
                    y.value.borrow().to_owned(),
                    z.value.borrow().to_owned(),
                )
            })
            .collect::<Vec<_>>());

        let mut nn = NN::new();
        nn.add_layer((2, 3), Value::relu);
        nn.add_layer((3, 1), Value::sigmoid);




    }
}
