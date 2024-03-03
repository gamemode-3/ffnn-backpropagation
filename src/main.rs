mod activation;
mod network;

use activation::Sigmoid;
use network::{Layer, LayerProperties, Network};

struct TestCase<const NUM_IN: usize, const NUM_OUT: usize> {
    input: [f64; NUM_IN],
    target: [f64; NUM_OUT],
}

fn main() {
    let mut network: Network<2, 1> = Network::random(
        vec![
            LayerProperties::new(3, Sigmoid),
            LayerProperties::new(2, Sigmoid),
        ],
        Sigmoid,
    );

    // let mut network: Network<2, 2> = Network::new(
    //     vec![
    //         Layer::new(
    //             vec![vec![0.15, 0.2], vec![0.25, 0.3]],
    //             vec![0.35, 0.35],
    //             Sigmoid,
    //         ),
    //         Layer::new(
    //             vec![vec![0.4, 0.45], vec![0.5, 0.55]],
    //             vec![0.6, 0.6],
    //             Sigmoid,
    //         ),
    //     ]
    // );

    let test_cases = [
        TestCase {
            input: [0.0, 0.0],
            target: [0.0],
        },
        TestCase {
            input: [0.0, 1.0],
            target: [1.0],
        },
        TestCase {
            input: [1.0, 0.0],
            target: [1.0],
        },
        TestCase {
            input: [1.0, 1.0],
            target: [0.0],
        },
    ];

    for _ in 0..10000 {
        for test_case in test_cases.iter() {
            network.backpropagate(test_case.input, test_case.target, 0.5);
        }
    }


    for test_case in test_cases.iter() {
        let output = network.forward(test_case.input);
        println!("{:?} -> {:?}", test_case.input, output);
    }
}
