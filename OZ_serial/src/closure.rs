use ndarray::Array1;

pub trait Closure {
    fn calculate(&self, r: &Array1<f64>, u: &Array1<f64>, t: &Array1<f64>, B: f64) -> Array1<f64>;
}

pub struct HyperNettedChain;

impl HyperNettedChain {
    pub fn new() -> Self {
        HyperNettedChain {}
    }
}

impl Closure for HyperNettedChain {
    fn calculate(&self, r: &Array1<f64>, u: &Array1<f64>, t: &Array1<f64>, B: f64) -> Array1<f64> {
        let mut exponent = -B * u + t;
        exponent.mapv_inplace(|a| a.exp());
        r * (exponent - 1.0 - t)
    }
}