use ndarray::{Array1, Array};
use std::sync::Arc;
use rustdct::{DctPlanner, Dst1};
use std::f64::consts::PI;
use plotly::common::Mode;
use plotly::{ImageFormat, Plot, Scatter};

pub mod potential;
pub mod integral;
pub mod closure;
pub mod solver;
pub mod data;
pub mod grid;

use crate::potential::{Potential, LennardJones};
use crate::closure::{Closure, HyperNettedChain};
use crate::integral::{IntegralEquation, OZInt};
use crate::data::Data;
use crate::grid::Grid;

pub struct OZ_builder<'a, P: Potential, C: Closure, I: IntegralEquation> {
    grid: Option<&'a Grid>,
    data: Option<Data>,
    pot: Option<P>,
    clos: Option<C>,
    int_eq: Option<I>
}

impl<'a, P: Potential, C: Closure, I: IntegralEquation> OZ_builder<'a, P, C, I> {
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

    pub fn int_eq(mut self, int_eq: I) -> Self {
        self.int_eq = Some(int_eq);
        self
    }

    pub fn build(self) -> OZ<'a, P, C, I> {
        OZ {
            grid: self.grid.expect("missing grid"),
            data: self.data.expect("missing data struct"),
            pot: self.pot.expect("missing potential"),
            clos: self.clos.expect("missing closure"),
            int_eq: self.int_eq.expect("missing intgral equation"),
            dft_plan: DctPlanner::new().plan_dst1(self.grid.unwrap().npts),
        }
    }
}

pub struct OZ<'a, P: Potential, C: Closure, I: IntegralEquation> {
    pub grid: &'a Grid,
    pub data: Data,
    pub pot: P,
    pub clos: C,
    pub int_eq: I,
    dft_plan: Arc<dyn Dst1<f64>>
}

impl<'a, P: Potential, C: Closure, I: IntegralEquation> OZ<'a, P, C, I> {

    pub fn build() -> OZ_builder<'a, P, C, I> {
        OZ_builder { grid: None, data: None, pot: None, clos: None, int_eq: None }
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
    
    fn operator(&self, cr: Array1<f64>, r: &Array1<f64>, k: &Array1<f64>, u: &Array1<f64>, p: f64, B: f64) -> Array1<f64> {
        // c(r) -> c(k)
        let ck = self.forward_dfbt(cr);

        let tk = self.int_eq.calculate(ck, k, p);
        // t(k) -> t(r)
        let tr = self.backward_dfbt(tk);

        // t(r) -> c_A(r)
        self.clos.calculate(r, u, &tr, B)
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
        let k = &self.grid.ki;
        let u = self.pot.calculate(r);
        let damp = 0.2;
        let tol = tol;
        let max_iter = maxiter;
        loop {
            // Store previous solution
            // Want a copy of it
            let c_prev = self.data.c.clone();
            //c_n -> c_A
            let c_A = self.operator(c_prev.clone(), r, k, &u, self.data.p, self.data.B);
            // Mixing step for new solution:
            // c_A -> c_n+1
            let c_new = &c_prev + (damp * (&c_A - &c_prev));

            println!("Iteration: {}, Diff: {:e}", i, (&c_new.sum() - &c_prev.sum()).abs());

            if (&c_new.sum() - &c_prev.sum()).abs() < tol {
                println!("converged");
                self.data.c = c_new.clone();
                let ck = self.forward_dfbt(c_new.clone());
                let tk = self.int_eq.calculate(ck, k, self.data.p);
                self.data.t = self.backward_dfbt(tk);
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
    let grid = Grid::new(2048, 10.24);

    // Set an initial guess for iteration
    let initial_guess: Array1<f64> = Array1::zeros(2048);

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
    .int_eq(OZInt::new())
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
    let scale = 10.0;
    // plot.write_html("out.html");
    plot.write_image(filename, image_format, width, height, scale);
}
