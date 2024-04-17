use crate::value::Value;

pub enum Activation {
    ReLU,
    Sigmoid,
}

impl Activation {
    pub fn apply(&self, value: &Value) -> Value {
        match self {
            Activation::ReLU => value.relu(),
            Activation::Sigmoid => value.sigmoid(),
        }
    }
}

struct Layer {
    weights: Vec<Vec<Value>>,
    biases: Vec<Value>,
    activation: Activation,
}

pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, shape: (usize, usize), activation: Activation) {
        let (input_size, output_size) = shape;
        if let Some(prev_layer) = self.layers.last() {
            assert_eq!(prev_layer.weights.len(), input_size);
        }
        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| Value::from(rand::random::<f32>() - 0.5))
                    .collect()
            })
            .collect();
        let biases = (0..output_size)
            .map(|_| Value::from(rand::random::<f32>() - 0.5))
            .collect();
        self.layers.push(Layer {
            weights,
            biases,
            activation,
        });
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.layers.iter().flat_map(|layer| {
            layer
                .weights
                .iter()
                .flat_map(|weights| weights.iter())
                .chain(layer.biases.iter())
        })
    }

    pub fn zero_grad(&self) {
        self.parameters().for_each(|param| param.set_grad(0.0));
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
                .map(|(value, bias)| (layer.activation.apply(&(&value + &bias))))
                .collect();
        }
        output
    }
}

#[cfg(test)]
mod tests {

    use plotly::{common::Mode, Plot, Scatter, Scatter3D};

    use super::*;

    #[test]
    fn test_nn() {
        let mut nn = NeuralNetwork::new();
        nn.add_layer((2, 3), Activation::ReLU);
        nn.add_layer((3, 1), Activation::Sigmoid);
        let input = vec![Value::from(1.0), Value::from(2.0)];
        let output = nn.forward(input).pop().unwrap();
        output.back_prop();
        output.print_graph();
    }

    #[test]
    fn small_dataset() {
        fn test_fn((x, y): (&Value, &Value)) -> Value {
            // let raw = &(&x.pow(2) + &y.pow(2)) - &Value::from(3.0);
            let raw = x;
            Value::from(if raw.value().is_sign_positive() {
                1.0
            } else {
                0.0
            })
        }

        // points around -3 to 3 in x and y
        let grid = (-3..=3)
            .flat_map(|x| (-3..=3).map(move |y| (Value::from(x as f32), Value::from(y as f32))))
            .collect::<Vec<_>>();
        let test_data = grid
            .iter()
            .map(|(x, y)| ((x, y), test_fn((x, y))))
            .collect::<Vec<_>>();

        let mut nn = NeuralNetwork::new();
        nn.add_layer((2, 5), Activation::ReLU);
        nn.add_layer((5, 1), Activation::Sigmoid);

        let mut plot = Plot::new();

        let x = test_data
            .iter()
            .map(|((x, _), _)| x.value())
            .collect::<Vec<_>>();
        let y = test_data
            .iter()
            .map(|((_, y), _)| y.value())
            .collect::<Vec<_>>();
        let z = test_data
            .iter()
            .map(|(_, target)| target.value())
            .collect::<Vec<_>>();
        let outputs = test_data
            .iter()
            .cloned()
            .map(|((x, y), _)| {
                let output = nn.forward(vec![x.clone(), y.clone()]).pop().unwrap();
                let output_value = output.value();
                output_value
            })
            .collect::<Vec<_>>();

        let trace = Scatter3D::new(x.clone(), y.clone(), z.clone())
            .mode(Mode::Markers)
            .name("Data");

        let trace2 = Scatter3D::new(x.clone(), y.clone(), outputs)
            .mode(Mode::Markers)
            .name("Untrained");
        plot.add_trace(trace);
        plot.add_trace(trace2);
        plot.write_html("test-data.html");

        const LEARNING_RATE: f32 = 0.01;
        const EPOCHS: usize = 900;

        let mut avg_losses: Vec<f32> = vec![];

        for _ in 0..EPOCHS {
            let losses = test_data
                .clone()
                .iter()
                .map(|((x, y), target)| {
                    let output = nn.forward(vec![(*x).clone(), (*y).clone()]).pop().unwrap();
                    let target = target.clone();
                    let loss = (&Value::from(1.0) - &(&output * &target)).relu();
                    loss
                })
                .collect::<Vec<_>>();

            let total_loss = &losses.into_iter().sum() * &Value::from(1.0 / test_data.len() as f32);
            nn.zero_grad();
            total_loss.back_prop();
            nn.parameters().for_each(|param| {
                param.set_value(param.value() - LEARNING_RATE * param.grad());
            });
            avg_losses.push(total_loss.value());
            dbg!(total_loss.value());
        }

        let outputs = test_data
            .iter()
            .cloned()
            .map(|((x, y), _)| {
                let output = nn.forward(vec![x.clone(), y.clone()]).pop().unwrap();
                let output_value = output.value();
                output_value
            })
            .collect::<Vec<_>>();

        let mut plot = Plot::new();
        let trace = Scatter3D::new(x.clone(), y.clone(), z.clone())
            .mode(Mode::Markers)
            .name("Data");

        let trace2 = Scatter3D::new(x, y, outputs)
            .mode(Mode::Markers)
            .name("Trained");
        plot.add_trace(trace);
        plot.add_trace(trace2);
        plot.write_html("test-data-trained.html");

        // plot losses over training
        let mut plot = Plot::new();
        let trace = Scatter::new((0..avg_losses.len()).collect::<Vec<_>>(), avg_losses)
            .mode(Mode::LinesMarkers)
            .name("Loss");
        plot.add_trace(trace);
        plot.write_html("test-losses.html");
    }
}
