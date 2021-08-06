use mathru::{
    algebra::linear::Vector,
    analysis::differential_equation::ordinary::{DormandPrince, ExplicitODE, ProportionalControl},
};
use num_traits;
use plotters::prelude::*;

use cons_laws::*;

#[inline]
#[allow(non_snake_case, unused_variables)]
fn V<T: num_traits::One>(t: T, x: T) -> T {
    T::one()
}

#[inline]
#[allow(non_snake_case, unused_variables)]
fn Wprime<T: num_traits::real::Real>(t: T, x: T) -> T {
    let one = T::one();
    let zero = T::zero();
    if x != zero {
        x.signum() / (one + x.abs())
    } else {
        zero
    }
}

#[inline]
#[allow(non_snake_case, unused_variables)]
fn W<T: num_traits::real::Real>(t: T, x: T) -> T {
    let one = T::one();
    (one + x.abs()).ln()
}

#[inline]
#[allow(non_snake_case, unused_variables)]
fn v<T: num_traits::real::Real>(rho: T) -> T {
    T::zero().max(T::one() - rho)
}

struct ConservationLaw<T, Vel, Int> {
    t_start: T,
    t_end: T,
    n: usize,
    vel: Vel,
    int: Int,
}

struct VectorIterator<'v, T> {
    v: &'v Vector<T>,
    i: usize,
}

impl<'v, T> VectorIterator<'v, T> {
    pub fn new(v: &'v Vector<T>) -> Self {
        Self { v, i: 0 }
    }
}

impl<'v, T> Iterator for VectorIterator<'v, T> {
    type Item = &'v T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.len() {
            let item = self.v.get(self.i);
            self.i += 1;
            Some(item)
        } else {
            None
        }
    }
}

impl<'v, T> ExactSizeIterator for VectorIterator<'v, T> {
    fn len(&self) -> usize {
        self.v.dim().0 * self.v.dim().1
    }
}

impl<T, Vel, Int> ExplicitODE<T> for ConservationLaw<T, Vel, Int>
where
    T: num_traits::real::Real + std::iter::Sum<T> + From<f64> + Clone + std::fmt::Debug,
    Vel: ExternalVelocity<T>,
    Int: Interaction<T>,
{
    fn func(&self, t: &T, x: &Vector<T>) -> Vector<T> {
        let t = *t;
        let n = self.n;
        let mut dx = x.clone();
        let f: T = ((n - 1) as f64).recip().into();
        for i in 0..n {
            let vel: T = self.vel.eval(t, *x.get(i));
            let int: T = self.int.eval(t, *x.get(i), VectorIterator::new(x).cloned());
            let tot = vel + int;
            let dens = if tot.is_sign_positive() {
                if i < n - 1 {
                    f / (*x.get(i + 1) - *x.get(i))
                } else {
                    T::zero()
                }
            } else {
                if i > 0 {
                    f / (*x.get(i) - *x.get(i - 1))
                } else {
                    T::zero()
                }
            };
            let m = v(dens);
            *dx.get_mut(i) = tot * m;
        }
        dx
    }
    fn time_span(&self) -> (T, T) {
        (self.t_start, self.t_end)
    }
    fn init_cond(&self) -> Vector<T> {
        let n = self.n;
        Vector::new_column(
            n,
            (0..n)
                .map(|i| ((i as f64) / ((n - 1) as f64)).into())
                .collect(),
        )
    }
}

fn main() -> Result<(), &'static str> {
    // let vel = Velocity::new(V);
    let sampl_inter = SampledInteraction::new(Wprime);
    let integ_inter = IntegratedInteraction::new(W);

    // Create a ODE solver instance
    let h_0: f64 = 0.0001;
    let fac: f64 = 0.9;
    let fac_min: f64 = 0.01;
    let fac_max: f64 = 2.0;
    let n_max: u32 = 200000;
    let abs_tol: f64 = 10e-8;
    let rel_tol: f64 = 10e-8;

    let solver: ProportionalControl<f64> =
        ProportionalControl::new(n_max, h_0, fac, fac_min, fac_max, abs_tol, rel_tol);

    let root_area = BitMapBackend::new("./figures/medium.png", (800, 600)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .margin(20)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Particle trajectories", ("Arial", 20))
        .build_cartesian_2d(0.0..5.0, 0.0..3.5)
        .unwrap();

    ctx.configure_mesh()
        .x_desc("time")
        .axis_desc_style(("sans-serif", 15).into_font())
        .y_desc("position")
        .axis_desc_style(("sans-serif", 15).into_font())
        .draw()
        .unwrap();

    // ODE with sampled interaction
    let problem = ConservationLaw {
        t_start: 0.0,
        t_end: 5.0,
        n: 80,
        vel: V,
        int: sampl_inter,
    };

    let clock_time = std::time::SystemTime::now();
    let (t, x): (Vec<f64>, Vec<Vector<f64>>) = solver.solve(&problem, &DormandPrince::default())?;

    println!("{:?}", clock_time.elapsed().unwrap());
    println!("{} steps", t.len());

    for i in 0..problem.n / 2 {
        let i = 2 * i;
        let mut graph: Vec<(f64, f64)> = Vec::with_capacity(t.len());

        for k in 0..x.len() {
            graph.push((t[k], *x[k].get(i)));
        }

        ctx.draw_series(LineSeries::new(graph, &BLUE)).unwrap();
    }

    // ODE with sampled interaction
    let problem = ConservationLaw {
        t_start: 0.0,
        t_end: 5.0,
        n: 80,
        vel: V,
        int: integ_inter,
    };

    let clock_time = std::time::SystemTime::now();
    let (t, x): (Vec<f64>, Vec<Vector<f64>>) = solver.solve(&problem, &DormandPrince::default())?;

    println!("{:?}", clock_time.elapsed().unwrap());
    println!("{} steps", t.len());

    for i in 0..problem.n / 2 {
        let i = 2 * i;
        let mut graph: Vec<(f64, f64)> = Vec::with_capacity(t.len());

        for k in 0..x.len() {
            graph.push((t[k], *x[k].get(i)));
        }

        ctx.draw_series(LineSeries::new(graph, &RED)).unwrap();
    }

    Ok(())
}
