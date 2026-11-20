use oxide_ml::prelude::*;

#[derive(Module)]
enum Kind {
    Two { first: ReLU, second: Tanh },
}

fn main() {}
