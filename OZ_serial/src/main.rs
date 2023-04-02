use ndarray::Array1;

pub trait Potential {
    fn calculate(&self, r: &Array1<f64>) -> Array1<f64>;
}
pub trait Closure {
    fn calculate(&self, u: &Array1<f64>, t: &Array1<f64>, B: f64) -> Array1<f64>;
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
        let ir12 = ir6.to_owned().clone();
        ir6.mapv_inplace(|a| a.powf(12.0));

        4.0 * self.epsilon * (ir12.to_owned() - ir6.to_owned())
    }
}

pub struct HyperNettedChain {}

impl Closure for HyperNettedChain {
    fn calculate(&self, u: &Array1<f64>, t: &Array1<f64>, B: f64) -> Array1<f64> {
        let mut exponent = -B * u + t;
        exponent.mapv_inplace(|a| a.exp());
        exponent - 1.0 - t
    }
}
pub enum PotentialType {}
pub enum ClosureType {}
pub struct OZ {
    data: data,
    pot: PotentialType,
    clos: ClosureType,
}
pub struct data {
    kT: f64,
    T: f64,
    p: f64,
    B: f64,
    u: Array1<f64>,
    c: Array1<f64>,
    h: Array1<f64>,
    t: Array1<f64>,
}

impl data { }

fn main() {}
