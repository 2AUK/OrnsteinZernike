use ndarray::{Array1, Array};
use rustdct::DctPlanner;


pub mod potential;
pub mod closure;
pub mod data;
pub mod grid;

use crate::potential::{Potential, LennardJones};
use crate::closure::{Closure, HyperNettedChain};
use crate::data::Data;
use crate::grid::Grid;

pub struct OZ<'a, P: Potential, C: Closure> {
    grid: &'a Grid,
    data: Data,
    pot: P,
    clos: C,
}

impl<'a, P: Potential, C: Closure> OZ<'a, P, C> {
    pub fn new(grid: &'a Grid, data: Data, pot: P, clos: C) -> Self {
        OZ {
            grid,
            data,
            pot,
            clos,
        }
    }
    
    fn int_eq(self) -> Array1<f64> {
        let c = self.data.c.view();
        let k = self.grid.ki.view();
        let p = self.data.p;

        let c2 = c.mapv(|a| a.powf(2.0));
        let pc2 = p * c2;
        let pc = p * &c;

        let inv_term = 1.0 / (&k - &pc);

        pc2 * inv_term
    }

    fn clean_up(mut self) {
        todo!();
    }
}


fn main() {
    let grid = Grid::new(1024, 10.24);
    let lj = LennardJones::new(3.4, 120.0);
    let T = 1.0;

    let r = &grid.ri.to_vec();
    let lj_r = (lj.calculate(&grid.ri) / T).to_vec();

    let problem = Data::build()
    .boltzmann_constant(1.0)
    .temperature(85.0)
    .density(0.021017479720736955)
    .npts(grid.npts)
    .build();

    println!("{:?}", &problem);

    let OZ_problem = OZ::new(&grid, problem, lj, HyperNettedChain::new());


}
