use ndarray::{Array1, Array};
use std::f64::consts::PI;
use std::sync::Arc;
use rustdct::TransformType4;

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

    pub fn dfbt(self, prefac: f64, func: &Array1<f64>, plan: Arc<dyn TransformType4<f64>>) -> Array1<f64> {
        let mut buffer = func.to_vec();
        plan.process_dst4(&mut buffer);
        prefac * Array1::from_vec(buffer)
    }
}