use ndarray::Array1;

pub trait IntegralEquation {
    fn calculate(&self, c: Array1<f64>, k: &Array1<f64>, p: f64) -> Array1<f64>;
}

pub struct OZInt;

impl OZInt {
    pub fn new() -> Self {
        OZInt {}
    }
}

impl IntegralEquation for OZInt {
    fn calculate(&self, ck: Array1<f64>, k: &Array1<f64>, p: f64) -> Array1<f64> {
        let c2 = ck.mapv(|a| a.powf(2.0));
        let pc = p * &ck;
        let pc2 = p * &c2;

        pc2 / (k - pc)
    } 
}