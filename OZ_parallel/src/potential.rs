use ndarray::Array1;

pub trait Potential {
    fn calculate(&self, r: &Array1<f64>) -> Array1<f64>;
}

pub struct LennardJones {
    pub sigma: f64,
    pub epsilon: f64,
}

impl LennardJones {
    pub fn new(sigma: f64, epsilon: f64) -> Self {
        LennardJones { sigma, epsilon }
    }
}

impl Potential for LennardJones {
    fn calculate(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut ir = self.sigma / r;
        let mut ir6 = ir.view_mut();
        ir6.mapv_inplace(|a| a.powf(6.0));
        let mut ir12 = ir6.to_owned().clone();
        ir12.mapv_inplace(|a| a.powf(2.0));

        4.0 * self.epsilon * (ir12.to_owned() - ir6.to_owned())
    }
}