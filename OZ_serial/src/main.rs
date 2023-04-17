use ndarray::{Array1, Array};
use std::sync::Arc;
use rustdct::{DctPlanner, Dst1};
use std::f64::consts::PI;
use plotly::common::Mode;
use plotly::ndarray::ArrayTraces;
use plotly::{ImageFormat, Plot, Scatter};

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
            dft_plan: DctPlanner::new().plan_dst1(self.grid.unwrap().npts),
        }
    }
}

pub struct OZ<'a, P: Potential, C: Closure> {
    pub grid: &'a Grid,
    pub data: Data,
    pub pot: P,
    pub clos: C,
    dft_plan: Arc<dyn Dst1<f64>>
}

impl<'a, P: Potential, C: Closure> OZ<'a, P, C> {

    pub fn build() -> OZ_builder<'a, P, C> {
        OZ_builder { grid: None, data: None, pot: None, clos: None }
    }

    pub fn initialise(mut self, initial_guess: Array1<f64>) -> Self {
        let pot = self.pot.calculate(&self.grid.ri);
        self.data.u = pot;
        self.data.c = initial_guess;
        self
    }

    fn forward_dfbt(&self, arr: Array1<f64>) -> Array1<f64> {
        let mut buffer = arr.to_vec();
        self.dft_plan.process_dst1(&mut buffer);
        2.0 * PI * self.grid.dr * Array1::from_vec(buffer)
    }

    fn backward_dfbt(&self, arr: Array1<f64>) -> Array1<f64> {
        let mut buffer = arr.to_vec();
        self.dft_plan.process_dst1(&mut buffer);
        self.grid.dk / (2.0 * PI).powf(2.0) * Array1::from_vec(buffer)
    }
    
    fn int_eq(&self) -> Array1<f64> {
        // TODO Implement the Fourier-Bessel transforms here
        let c = self.data.c.view();
        let k = self.grid.ki.view();
        let p = self.data.p;

        // c(r) -> c(k)
        let ck = self.forward_dfbt(c.clone().to_owned());
        let c2 = ck.mapv(|a| a.powf(2.0));
        let pc2 = p * c2;
        let pc = p * &ck;

        let inv_term = 1.0 / (&k - &pc);

        // t(k) -> t(r)
        self.backward_dfbt(pc2 * inv_term)
    }

    fn clean_up(mut self) -> Self {
        self.data.c = &self.data.c / &self.grid.ri;
        self.data.t = &self.data.t / &self.grid.ri;
        self.data.h = &self.data.t + &self.data.c;

        self
    }

    pub fn solve(mut self, tol: f64, maxiter: u32) -> Self {
        let mut i = 0;
        let r = &self.grid.ri;
        let u = self.pot.calculate(r);
        let damp = 0.2;
        let tol = tol;
        let max_iter = maxiter;
        loop {
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
            let c_new = &c_prev + (damp * (&c_A - &c_prev));

            println!("Iteration: {}, Diff: {:e}", i, (&c_new.sum() - &c_prev.sum()).abs());

            if (&c_new.sum() - &c_prev.sum()).abs() < tol {
                println!("converged");
                self.data.c = c_new;
                self.data.t = t;
                self.data.h = &self.data.t + &self.data.c;
                break;
            }

            i += 1;

            if i == max_iter {
                println!("max iteration reached");
                break;
            }

            self.data.c = c_new;

        }
        self
    }
}


fn main() {
    // Initialise grid
    let grid = Grid::new(16384, 10.24);

    // Set an initial guess for iteration
    let initial_guess: Array1<f64> = Array1::zeros(16384);

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
    .initialise(initial_guess)
    .solve(1e-5, 10000)
    .clean_up();

    let gr = 1.0 + OZ_method.data.h;

    let trace = Scatter::from_array(OZ_method.grid.ri.clone(), gr).mode(Mode::LinesMarkers);
    let mut plot = Plot::new();
    plot.add_trace(trace);
    
    let filename = "out";
    let image_format = ImageFormat::PNG;
    let width = 800;
    let height = 600;
    let scale = 1.0;

    plot.write_image(filename, image_format, width, height, scale);
}
