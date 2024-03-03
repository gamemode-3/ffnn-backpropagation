mod activation;
mod layer;

use activation::Sigmoid;
use layer::Layer;

fn main() {
    let network = Layer::<4, 100>::random(Sigmoid)
        .with_next::<1000>(Layer::random(Sigmoid)
            .with_next::<2>(Layer::random(Sigmoid)
                .with_next::<4>(Layer::random(Sigmoid))
            ),
        );

    println!("network created");

    let input = [1.0, -2.0, -3.0, -4.0];

    let output = network.run(input);

    println!("{:?}", output);
}
