use mathru::{
    algebra::linear::Vector,
    analysis::differential_equation::ordinary::{DormandPrince, ExplicitODE, ProportionalControl},
};
use num_traits;
use plotters::prelude::*;

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
fn v<T: num_traits::real::Real>(rho: T) -> T {
    T::zero().max(T::one() - rho)
}

struct ConservationLaw<T: num_traits::real::Real> {
    t_start: T,
    t_end: T,
    n: usize,
}

impl<T> ExplicitODE<T> for ConservationLaw<T>
where
    T: num_traits::real::Real + std::iter::Sum<T> + From<f64>,
{
    fn func(&self, t: &T, x: &Vector<T>) -> Vector<T> {
        let t = *t;
        let n = self.n;
        let mut dx = x.clone();
        let f: T = ((n - 1) as f64).recip().into();
        for i in 0..n {
            let ext: T = V(t, *x.get(i));
            let int: T = -(0..self.n)
                .map(|j| Wprime(t, *x.get(i) - *x.get(j)))
                .sum::<T>()
                / (((n - 1) as f64).into());
            let vel = ext + int;
            let dens = if vel.is_sign_positive() {
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
            *dx.get_mut(i) = vel * m;
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
    // Create the ODE problem
    let problem = ConservationLaw::<f64> {
        t_start: 0.0,
        t_end: 5.0,
        n: 80,
    };

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

    let (t, x): (Vec<f64>, Vec<Vector<f64>>) = solver.solve(&problem, &DormandPrince::default())?;

    println!("{} steps", t.len());

    let root_area = BitMapBackend::new("./figures/basic.png", (800, 600)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .margin(20)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Particle trajectories", ("Arial", 20))
        .build_cartesian_2d(problem.t_start..problem.t_end, 0.0f64..3.5f64)
        .unwrap();

    ctx.configure_mesh()
        .x_desc("time")
        .axis_desc_style(("sans-serif", 15).into_font())
        .y_desc("position")
        .axis_desc_style(("sans-serif", 15).into_font())
        .draw()
        .unwrap();

    for i in 0..problem.n / 2 {
        let i = 2 * i;
        let mut graph: Vec<(f64, f64)> = Vec::with_capacity(t.len());

        for k in 0..x.len() {
            graph.push((t[k], *x[k].get(i)));
        }

        ctx.draw_series(LineSeries::new(graph, &BLACK)).unwrap();
    }

    Ok(())
}
