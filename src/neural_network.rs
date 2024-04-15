use crate::value::Value;

struct Layer {
    weights: Vec<Vec<Value>>,
    biases: Vec<Value>,
    activation: fn(Value) -> Value,
}

pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
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
                    .map(|_| Value::new(rand::random::<f32>() - 0.5, None, None))
                    .collect()
            })
            .collect();
        let biases = (0..output_size)
            .map(|_| Value::new(rand::random::<f32>() - 0.5, None, None))
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

    use plotly::{common::Mode, Plot, Scatter, Scatter3D};

    use super::*;

    #[test]
    fn test_nn() {
        let mut nn = NeuralNetwork::new();
        nn.add_layer((2, 3), Value::relu);
        nn.add_layer((3, 1), Value::relu);
        let input = vec![Value::new(1.0, None, None), Value::new(2.0, None, None)];
        let output = nn.forward(input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn small_dataset() {
        fn test_fn((x, y): (&Value, &Value)) -> Value {
            let raw = &(&x.pow(2) + &y.pow(2)) - &Value::new(3.0, None, None);
            Value::new(
                if raw.value.borrow().is_sign_positive() {
                    1.0
                } else {
                    0.0
                },
                None,
                None,
            )
        }

        // points around -3 to 3 in x and y
        let grid = (-3..=3)
            .flat_map(|x| {
                (-3..=3).map(move |y| (Value::new(x as f32, None, None), Value::new(y as f32, None, None)))
            })
            .collect::<Vec<_>>();
        let test_data = grid
            .iter()
            .map(|(x, y)| ((x, y), test_fn((x, y))))
            .collect::<Vec<_>>();

        dbg!(test_data
            .iter()
            .map(|((x, y), z)| { (x.value.to_owned(), y.value.to_owned(), z.value.to_owned(),) })
            .collect::<Vec<_>>());

        let mut nn = NeuralNetwork::new();
        nn.add_layer((2, 10), Value::relu);
        nn.add_layer((10, 5), Value::relu);
        nn.add_layer((5, 1), Value::sigmoid);

        let mut plot = Plot::new();

        let x = test_data
            .iter()
            .map(|((x, _), _)| *x.value.borrow())
            .collect::<Vec<_>>();
        let y = test_data
            .iter()
            .map(|((_, y), _)| *y.value.borrow())
            .collect::<Vec<_>>();
        let z = test_data
            .iter()
            .map(|(_, target)| *target.value.borrow())
            .collect::<Vec<_>>();
        let outputs = test_data
            .iter()
            .cloned()
            .map(|((x, y), _)| {
                let output = nn.forward(vec![x.clone(), y.clone()])[0].clone();
                let output_value = *output.value.borrow();
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

        // do optimization by hand
        fn loss(output: &Value, target: &Value) -> Value {
            let diff = target - output;
            diff.pow(2)
        }

        const LEARNING_RATE: f32 = 0.05;
        const EPOCHS: usize = 300;

        let mut avg_losses: Vec<f32> = vec![];

        for _ in 0..EPOCHS {
            let mut losses = vec![];
            for ((x, y), target) in test_data.iter().cloned() {
                let output = nn.forward(vec![x.clone(), y.clone()])[0].clone();
                let mut loss = loss(&output, &target);
                loss.back_prop();
                losses.push(*loss.value.borrow());
                nn.parameters().for_each(|param| {
                    *param.value.borrow_mut() -= LEARNING_RATE * *param.grad.borrow();
                });
            }
            avg_losses.push(losses.iter().sum::<f32>() / losses.len() as f32);
        }

        // plot outputs after training
        let outputs = test_data
            .iter()
            .cloned()
            .map(|((x, y), _)| {
                let output = nn.forward(vec![x.clone(), y.clone()])[0].clone();
                let output_value = *output.value.borrow();
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
