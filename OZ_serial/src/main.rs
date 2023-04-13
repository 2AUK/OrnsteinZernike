use ndarray::{Array1, Array};
use rustdct::DctPlanner;


pub mod potential;
pub mod closure;
pub mod solver;
pub mod data;
pub mod grid;

use crate::potential::{Potential, LennardJones};
use crate::closure::{Closure, HyperNettedChain};
use crate::data::Data;
use crate::grid::Grid;

pub struct OZ_builder<'a, P: Potential, C: Closure> {
    grid: Option<&'a Grid>,
    data: Option<Data>,
    pot: Option<P>,
    clos: Option<C>,
}

impl<'a, P: Potential, C: Closure> OZ_builder<'a, P, C> {
    pub fn data(mut self, data: Data) -> Self {
        self.data = Some(data);
        self
    }
    
    pub fn grid(mut self, grid: &'a Grid) -> Self {
        self.grid = Some(grid);
        self
    }

    pub fn pot(mut self, pot: P) -> Self {
        self.pot = Some(pot);
        self
    }

    pub fn clos(mut self, clos: C) -> Self {
        self.clos = Some(clos);
        self
    }

    pub fn build(self) -> OZ<'a, P, C> {
        OZ {
            grid: self.grid.expect("missing grid"),
            data: self.data.expect("missing data struct"),
            pot: self.pot.expect("missing potential"),
            clos: self.clos.expect("missing closure"),
        }
    }
}

pub struct OZ<'a, P: Potential, C: Closure> {
    pub grid: &'a Grid,
    pub data: Data,
    pub pot: P,
    pub clos: C,
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

    pub fn build() -> OZ_builder<'a, P, C> {
        OZ_builder { grid: None, data: None, pot: None, clos: None }
    }

    pub fn initialise(mut self, initial_guess: Array1<f64>) -> Self {
        let pot = self.pot.calculate(&self.grid.ri);
        self.data.u = pot;
        self.data.c = initial_guess;
        self
    }
    
    fn int_eq(&self) -> Array1<f64> {
        // TODO Implement the Fourier-Bessel transforms here
        let c = self.data.c.view();
        let k = self.grid.ki.view();
        let p = self.data.p;

        // c(r) -> c(k)
        let c2 = c.mapv(|a| a.powf(2.0));
        let pc2 = p * c2;
        let pc = p * &c;

        let inv_term = 1.0 / (&k - &pc);

        pc2 * inv_term
        // t(k) -> t(r)
    }

    fn clean_up(mut self) {
        self.data.c = self.data.c / &self.grid.ri;
        self.data.t = self.data.t / &self.grid.ri;
        self.data.h = self.data.t + self.data.c;
    }

    pub fn solve(mut self, tol: f64, maxiter: u32) {
        let i = 0;
        let r = &self.grid.ri;
        let u = self.pot.calculate(r);
        'iter: loop {
            // Store previous solution
            // Want a copy of it
            let c_prev = self.data.c.clone();
            
            // OZ operator:
            // c -> t -> c_A -> c_new
            // OZ equation:
            // c -> t
            let t = self.int_eq();
            // Closure:
            // t -> c_A
            let c_A = self.clos.calculate(r, &u, &t, self.data.B);
            // Mixing step for new solution:
            // c_A -> c_new
            let c_new = todo!();
        }
    }
}


fn main() {
    // Initialise grid
    let grid = Grid::new(1024, 10.24);

    // Set an initial guess for iteration
    let initial_guess: Array1<f64> = Array1::zeros(1024);

    // Construct the data required for the problem
    let problem = Data::build()
    .boltzmann_constant(1.0)
    .temperature(85.0)
    .density(0.021017479720736955)
    .npts(grid.npts)
    .build();

    // Construct the algorithm for the problem and initialise the data
    let OZ_method = OZ::build()
    .grid(&grid)
    .pot(LennardJones::new(3.4, 120.0))
    .clos(HyperNettedChain::new())
    .data(problem)
    .build()
    .initialise(initial_guess);

}
