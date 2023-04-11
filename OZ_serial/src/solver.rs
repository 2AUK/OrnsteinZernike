use ndarray::Array1;

pub trait ConvProblem {
    fn update(&mut self, input: &Array1<f64>) -> Array1<f64>;

    fn residual(&mut self, input: &Array1<f64>, output: &Array1<f64>) -> f64;
}

pub trait Converger {
    fn next_iter();
}