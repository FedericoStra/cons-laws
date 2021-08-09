use ode_solvers::SVector;
use plotters::prelude::*;

use cons_laws::*;

const N: usize = 201;

fn main() -> Result<(), &'static str> {
    use ode_solvers::dopri5::*;

    // let vel = Velocity::new(V);
    let sampl_inter = SampledInteraction::new(Wprime);
    let integ_inter = IntegratedInteraction::new(W);

    let system = SingleSpeciesModel::new(V, sampl_inter, v);

    let t_start = 0.0;
    let t_end = 15.0;
    let dt = 0.001;

    let x0 =
        SVector::<f64, N>::from_iterator((0..N).map(|i| 3.0 * i as f64 / (N - 1) as f64 - 1.5));

    let rtol = 1e-10;
    let atol = 1e-10;

    let mut stepper = Dopri5::new(system, t_start, t_end, dt, x0, rtol, atol);

    let clock = std::time::Instant::now();
    let res = stepper.integrate();
    let elapsed = clock.elapsed();

    let t = stepper.x_out();
    let x = stepper.y_out();
    let n_steps = t.len();

    println!("n_steps = {}\ntime = {:?}", n_steps, elapsed);
    println!("res = {:?}", res);

    let root_area =
        BitMapBackend::new("./figures/traffic_ode_solver.png", (800, 600)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .margin(20)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Particle trajectories", ("Arial", 20))
        .build_cartesian_2d(t_start..t_end, -1.5..7.0)
        .unwrap();

    ctx.configure_mesh()
        .x_desc("time")
        .axis_desc_style(("sans-serif", 15).into_font())
        .y_desc("position")
        .axis_desc_style(("sans-serif", 15).into_font())
        .draw()
        .unwrap();

    for i in 0..N / 10 {
        let i = 10 * i;
        let mut graph: Vec<(f64, f64)> = Vec::with_capacity(t.len());

        for k in 0..t.len() {
            if k % 5 == 0 {
                graph.push((t[k], x[k][i]));
            }
        }

        ctx.draw_series(LineSeries::new(graph, &BLUE)).unwrap();
    }
    ctx.draw_series(LineSeries::new([], &BLACK))
        .unwrap()
        .label("sampled")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    let system = SingleSpeciesModel::new(V, integ_inter, v);

    let t_start = 0.0;
    let t_end = 15.0;
    let dt = 0.001;

    // let x0 = SVector::from_iterator((0..N).map(|i| 3.0 * i as f64 / (N - 1) as f64 - 1.5));

    let rtol = 1e-10;
    let atol = 1e-10;

    let mut stepper = Dopri5::new(system, t_start, t_end, dt, x0, rtol, atol);

    let clock = std::time::Instant::now();
    let res = stepper.integrate();
    let elapsed = clock.elapsed();

    let t = stepper.x_out();
    let x = stepper.y_out();
    let n_steps = t.len();

    println!("n_steps = {}\ntime = {:?}", n_steps, elapsed);
    println!("res = {:?}", res);

    for i in 0..N / 10 {
        let i = 10 * i;
        let mut graph: Vec<(f64, f64)> = Vec::with_capacity(t.len());

        for k in 0..t.len() {
            if k % 5 == 0 {
                graph.push((t[k], x[k][i]));
            }
        }

        ctx.draw_series(LineSeries::new(graph, &RED)).unwrap();
    }
    ctx.draw_series(LineSeries::new([], &BLACK))
        .unwrap()
        .label("integrated")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    ctx.configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .border_style(&BLACK)
        .draw()
        .unwrap();

    Ok(())
}

#[inline]
#[allow(non_snake_case, unused_variables)]
fn V<T: num_traits::real::Real>(t: T, x: T) -> T {
    let sign = if t < T::from(7.5).unwrap() {
        T::one()
    } else {
        -T::one()
    };
    let magn = T::from(2).unwrap()
        + T::from(0.5).unwrap() * (x + x).sin()
        + T::from(0.3).unwrap() * (T::from(2.0 * (2.0_f64).sqrt()).unwrap() * x).sin()
        + T::from(0.2).unwrap() * (T::from(2.0 * (7.0_f64).sqrt()).unwrap() * x).cos();
    sign * magn
}

#[inline]
#[allow(non_snake_case, unused_variables)]
fn Wprime<T: num_traits::real::Real>(t: T, x: T) -> T {
    let one = T::one();
    let zero = T::zero();
    if x != zero {
        (x + x - x.signum()) * (T::one() + t.sin())
    } else {
        zero
    }
}

#[inline]
#[allow(non_snake_case, unused_variables)]
fn W<T: num_traits::real::Real>(t: T, x: T) -> T {
    (x * x - x.abs()) * (T::one() + t.sin())
}

#[inline]
#[allow(non_snake_case, unused_variables)]
fn v<T: num_traits::real::Real>(t: T, rho: T) -> T {
    T::zero().max(T::one() - rho)
}
