use ndarray::Array1;

#[derive(Debug)]
pub struct Data {
    pub kT: f64,
    pub T: f64,
    pub p: f64,
    pub B: f64,
    pub u: Array1<f64>,
    pub c: Array1<f64>,
    pub h: Array1<f64>,
    pub t: Array1<f64>,
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