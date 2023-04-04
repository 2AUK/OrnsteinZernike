use ndarray::{Array1, Array};
use plotly::common::Title;
use plotly::layout::{Axis, Layout};
use plotly::{Plot, Scatter};
use std::f64::consts::PI;

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
        let mut ir12 = ir6.to_owned().clone();
        ir12.mapv_inplace(|a| a.powf(2.0));

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
pub struct OZ<'a> {
    grid: &'a Grid,
    data: Data,
    pot: PotentialType,
    clos: ClosureType,
}
pub struct Grid {
    pub npts: usize,
    pub radius: f64,
    pub dr: f64,
    pub dk: f64,
    pub ri: Array1<f64>,
    pub ki: Array1<f64>,
}
impl Grid {
    pub fn new(npts: usize, radius: f64) -> Self {
        let dr: f64 = radius / npts as f64;
        let dk: f64 = 2.0 * PI / (2.0 * npts as f64 * dr);
        let ri: Array1<f64> = Array::range(0.5, npts as f64, 1.0) * dr;
        let ki: Array1<f64> = Array::range(0.5, npts as f64, 1.0) * dk;

        Grid {
            npts,
            radius,
            dr,
            dk,
            ri,
            ki,
        }
    }
}
#[derive(Debug)]
pub struct Data {
    kT: f64,
    T: f64,
    p: f64,
    B: f64,
    u: Array1<f64>,
    c: Array1<f64>,
    h: Array1<f64>,
    t: Array1<f64>,
}

impl Data { 
    pub fn build() -> DataBuilder {
        DataBuilder {
            kT: None,
            T: None,
            p: None,
            npts: None,
        }
    }
}

pub struct DataBuilder {
    kT: Option<f64>,
    T: Option<f64>,
    p: Option<f64>,
    npts: Option<usize>,
}

impl DataBuilder {
    pub fn boltzmann_constant(mut self, kT: f64) -> Self {
        self.kT = Some(kT);
        self
    }

    pub fn temperature(mut self, T: f64) -> Self {
        self.T = Some(T);
        self
    }

    pub fn density(mut self, p: f64) -> Self {
        self.p = Some(p);
        self
    }

    pub fn npts(mut self, npts: usize) -> Self {
        self.npts = Some(npts);
        self
    }

    pub fn build(self) -> Data {
        let npts = self.npts.expect("missing npts; required for defining grid");
        let kT =  self.kT.expect("missing Boltzmann constant; required for problem");
        let T = self.T.expect("missing temperature; required for problem");
        let p = self.p.expect("missing density; required for problem");
        let B = 1.0 / kT / T;
        let u = Array1::<f64>::zeros(npts);
        let c = Array1::<f64>::zeros(npts);
        let h = Array1::<f64>::zeros(npts);
        let t = Array1::<f64>::zeros(npts);
        Data {
            kT,
            T,
            p,
            B,
            u,
            c,
            h,
            t,
        }
    }
}

fn main() {
    let grid = Grid::new(16384, 10.24);
    let lj = LennardJones::new(3.16, 78.15);
    let T = 1.0;

    let r = &grid.ri.to_vec();
    let lj_r = (lj.calculate(&grid.ri) / T).to_vec();

    let problem = Data::build()
    .boltzmann_constant(1.0)
    .temperature(300.0)
    .density(0.2)
    .npts(1024)
    .build();

    println!("{:?}", problem);
}
