//! Deterministic particle methods for 1D conservation laws.
//!
//! This crate implements the deterministic particle schemes described in the article
//! *Entropy solutions of non-local scalar conservation laws with congestion via deterministic particle method*,
//! E. Radici, F. Stra, (2021), [https://arxiv.org/abs/2107.10760](https://arxiv.org/abs/2107.10760).
//!
//! You can cite the article as
//!
//! ```text
//! @online{RadiciStra2021,
//!     title={Entropy solutions of non-local scalar conservation laws with congestion via deterministic particle method},
//!     author={Emanuela Radici and Federico Stra},
//!     year={2021},
//!     eprint={2107.10760},
//!     archivePrefix={arXiv},
//!     primaryClass={math.AP},
//!     url={https://arxiv.org/abs/2107.10760}
//! }
//! ```
//!
//! This is a reimplementation in Rust of the Julia package
//! [ConservationLawsParticles.jl](https://github.com/FedericoStra/ConservationLawsParticles.jl).
//!
//! # What's the goal?
//!
//! The conservation law to be solved is
//!
//! $$
//! \partial_t\rho(t,x) + \mathop{\mathrm{div}}_x\bigl[\rho(t,x)v\bigl(\rho(t,x)\bigr)\bigl(V(t,x)-(\partial_xW*\rho)(t,x)\bigr)\bigr] = 0 ,
//! $$
//!
//! where
//!
//! - $V:[0,\infty)\times\mathbb{R}\to\mathbb{R}$ is the external velocity,
//! - $W:[0,\infty)\to\mathbb{R}$ is the interaction potential,
//! - $v:[0,\infty)\to[0,\infty)$ is a non increasing function to model congestion,
//! - the convolution is performed in space only.
//!
//! The approach to solve the conservation law is by tracking the motion of $N+1$ particles
//! $X=(x_0,x_1,\dots,x_N)$ such that between each consecutive pair of particles there is
//! a fraction $1/N$ of the total mass.
//!
//! Two deterministic particle schemes are described: one with integrated interaction $(\mathrm{ODE}_I)$ and one with sampled interaction $(\mathrm{ODE}_S)$:
//!
//! ```math
//! \begin{aligned}
//! (\mathrm{ODE}_I): & \left\{\begin{aligned}
//! x_i'(t) &= v_i(t) \bar U_i(t), \\
//! \bar U_i(t) &= V\bigl(t,x_i(t)\bigr) - (W*\partial_x\bar\rho)\bigl(t,x_i(t)\bigr)
//!     = V\bigl(t,x_i(t)\bigr) - \sum_{j=0}^N(\rho_{j+1}(t) - \rho_j(t)) W\bigl(t,x_i(t)-x_j(t)\bigr), \\
//! v_i(t) &= \begin{cases}
//!     v\bigl(\rho_i(t)\bigr), & \text{if } \bar U_i(t) < 0, \\
//!     v\bigl(\rho_{i+1}(t)\bigr), & \text{if } \bar U_i(t) \geq 0, \end{cases}
//! \end{aligned}\right. \\
//! (\mathrm{ODE}_S): & \left\{\begin{aligned}
//! x_i'(t) &= v_i(t) \dot U_i(t), \\
//! \dot U_i(t) &= V\bigl(t,x_i(t)\bigr) - (\partial_xW*\dot\rho)\bigl(t,x_i(t)\bigr)
//!     = V\bigl(t,x_i(t)\bigr) - \frac1N\sum_{j=0}^NW'\bigl(t,x_i(t)-x_j(t)\bigr), \\
//! v_i(t) &= \begin{cases}
//!     v\bigl(\rho_i(t)\bigr), & \text{if } \dot U_i(t) < 0, \\
//!     v\bigl(\rho_{i+1}(t)\bigr), & \text{if } \dot U_i(t) \geq 0. \end{cases}
//! \end{aligned}\right.
//! \end{aligned}
//! ```
//!
//! Here $\rho_i$ denotes the quantity $\frac1{N(x_i-x_{i-1})}$,
//! $\bar\rho$ is the piecewise constant density $\sum_{i=1}^N \rho_i 1_{[x_{i-1},x_i]}$
//! and $\dot\rho$ is the atomic measure $\frac1N \sum_{i=0}^N \delta_{x_i}$.
//! Notice that $\dot\rho$ is not a probability.
//!
//! # Example
//!
//! This example requires the `ode_solver` feature.
//!
//! Consider the following external velocity, interaction potential and mobility:
//!
//! ```math
//! \begin{aligned}
//! V(t,x) &= \mathop{\mathrm{sign}}(7.5-t)
//!     [2 + 0.5\sin(2x) + 0.3\sin(2\sqrt2x) + 0.2\cos(2\sqrt7x)] , \\
//! W(t,x) &= (1 + \sin(t)) (x^2 - |x|) , \\
//! v(t,\rho) &= (1-\rho)_+ .
//! \end{aligned}
//! ```
//!
//! The initial datum $\rho_0 = 1_{[-1.5,1.5]}$ is discretized $N=201$ with equally spaced particles.
//!
#![cfg_attr(feature = "ode_solver", doc = "```no_run")]
#![cfg_attr(not(feature = "ode_solver"), doc = "```ignore")]
//! use cons_laws::*;
//! use ode_solvers::{dopri5::Dopri5, SVector};
//!
//! #[inline]
//! fn V(t: f64, x: f64) -> f64 {
//!     let sign = if t < 7.5 { 1.0 } else { -1.0 };
//!     let magn = 2.0 + 0.5 * (2.0 * x).sin() + 0.3 * (2.0 * 2.0f64.sqrt() * x).sin()
//!         + 0.2 * (2.0 * 7.0f64.sqrt() * x).cos();
//!     sign * magn
//! }
//! #[inline]
//! fn Wprime(t: f64, x: f64) -> f64 { (x + x - x.signum()) * (1.0 + t.sin()) }
//! #[inline]
//! fn W(t: f64, x: f64) -> f64 { (x * x - x.abs()) * (1.0 + t.sin()) }
//! #[inline]
//! fn v(t: f64, rho: f64) -> f64 { 0.0f64.max(1.0 - rho) }
//!
//! // define the model, using the sampled or integrated interaction
//! let sampl_inter = SampledInteraction::new(Wprime);
//! let integ_inter = SampledInteraction::new(W);
//! let model = SingleSpeciesModel::new(V, sampl_inter, v);
//!
//! // initial condition
//! const N: usize = 201;
//! let x0 = SVector::<f64, N>::from_iterator((0..N).map(|i| 3.0 * i as f64 / (N - 1) as f64 - 1.5));
//!
//! // configure the solver
//! let t_start = 0.0;
//! let t_end = 15.0;
//! let dt = 0.001;
//! let rtol = 1e-10;
//! let atol = 1e-10;
//! let mut solver = Dopri5::new(model, t_start, t_end, dt, x0, rtol, atol);
//!
//! // solve the system of ODEs
//! solver.integrate();
//! let t = solver.x_out(); // times
//! let x = solver.y_out(); // particles positions
//! let n_steps = t.len();
//! ```
//! Plotting (some of) the trajectories with [`plotters`](https://crates.io/crates/plotters)
//! produces the following picture:
//!
//! ![Traffic with `ode_solver` feature][traffic_ode_solver]
//!
#![doc = ::embed_doc_image::embed_image!("traffic_ode_solver", "doc/imgs/traffic_ode_solver.png")]
#![cfg_attr(docsrs, feature(doc_cfg))]

/// A time-dependent velocity field that affects the particles.
pub trait ExternalVelocity<T, X = T> {
    fn eval(&self, t: T, x: X) -> X;
}

/// A time-dependent interaction between the particles.
pub trait Interaction<T, X = T> {
    fn eval<P>(&self, t: T, x: X, p: P) -> X
    where
        P: IntoIterator<Item = X>;
}

/// A time-dependent mobility.
pub trait Mobility<T, X = T> {
    fn eval(&self, t: T, x: X) -> X;
}

mod _zero;
pub use _zero::{OneMobility, ZeroInteraction, ZeroVelocity};

pub struct Velocity<V>(V);

impl<V> Velocity<V> {
    #[allow(non_snake_case)]
    pub fn new(V: V) -> Self {
        Self(V)
    }
}

impl<T, X, V> ExternalVelocity<T, X> for Velocity<V>
where
    V: Fn(T, X) -> X,
{
    #[inline]
    fn eval(&self, t: T, x: X) -> X {
        self.0(t, x)
    }
}

impl<T, X, V> ExternalVelocity<T, X> for V
where
    V: Fn(T, X) -> X,
{
    #[inline]
    fn eval(&self, t: T, x: X) -> X {
        self(t, x)
    }
}

impl<T, X, F> Mobility<T, X> for F
where
    F: Fn(T, X) -> X,
{
    #[inline]
    fn eval(&self, t: T, x: X) -> X {
        self(t, x)
    }
}

pub struct SampledInteraction<Wprime>(Wprime);

impl<Wprime> SampledInteraction<Wprime> {
    #[allow(non_snake_case)]
    pub fn new(Wprime: Wprime) -> Self {
        Self(Wprime)
    }
}

impl<T, X, Wprime> Interaction<T, X> for SampledInteraction<Wprime>
where
    T: Copy,
    X: Copy + num_traits::real::Real,
    Wprime: Fn(T, X) -> X,
{
    fn eval<P>(&self, t: T, x: X, p: P) -> X
    where
        P: IntoIterator<Item = X>,
    {
        let mut sum = X::zero();
        let mut len = 0;
        for y in p.into_iter() {
            len += 1;
            let r = x - y;
            // The function `self.0` (which is `W'`) must return `0` if `y == x`.
            // We can ensure it here by skipping the `x` when we encounter it.
            if r != X::zero() {
                sum = sum + self.0(t, r);
            }
        }
        -X::from(sum).unwrap() / X::from(len - 1).unwrap()
    }
}

pub struct IntegratedInteraction<W>(W);

impl<W> IntegratedInteraction<W> {
    #[allow(non_snake_case)]
    pub fn new(W: W) -> Self {
        Self(W)
    }
}

impl<T, X, W> Interaction<T, X> for IntegratedInteraction<W>
where
    T: Copy,
    X: Copy + num_traits::real::Real,
    W: Fn(T, X) -> X,
{
    fn eval<P>(&self, t: T, x: X, p: P) -> X
    where
        P: IntoIterator<Item = X>,
    {
        let mut p = p.into_iter().peekable();
        let mut total = X::zero();
        let mut left_dens = X::zero();
        let mut len = 0;
        while let Some(y) = p.next() {
            len += 1;
            let right_dens = if let Some(z) = p.peek() {
                (*z - y).recip()
            } else {
                X::zero()
            };
            total = total + self.0(t, x - y) * (right_dens - left_dens);
            left_dens = right_dens;
        }
        -total / X::from(len - 1).unwrap()
    }
}

pub struct SingleSpeciesModel<T, X, Vel, Int, Mob> {
    pub vel: Vel,
    pub int: Int,
    pub mob: Mob,
    _t: std::marker::PhantomData<T>,
    _x: std::marker::PhantomData<X>,
}

impl<T, X, Vel, Int, Mob> SingleSpeciesModel<T, X, Vel, Int, Mob>
where
    Vel: ExternalVelocity<T, X>,
    Int: Interaction<T, X>,
    Mob: Mobility<T, X>,
{
    pub fn new(vel: Vel, int: Int, mob: Mob) -> Self {
        Self {
            vel,
            int,
            mob,
            _t: std::marker::PhantomData,
            _x: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "ode_solver")]
pub use _ode_solver::*;

#[cfg(feature = "ode_solver")]
#[cfg_attr(docsrs, doc(cfg(feature = "ode_solver")))]
mod _ode_solver {
    use super::*;

    use ode_solvers::{SVector, System};

    impl<X, Vel, Int, Mob, const N: usize> System<SVector<X, N>>
        for SingleSpeciesModel<f64, X, Vel, Int, Mob>
    where
        Vel: ExternalVelocity<f64, X>,
        Int: Interaction<f64, X>,
        Mob: Mobility<f64, X>,
        X: num_traits::real::Real + nalgebra::base::Scalar,
    {
        fn system(&self, t: f64, x: &SVector<X, N>, dx: &mut SVector<X, N>) {
            let n = x.len();
            let f = X::from(n - 1).unwrap().recip();
            for i in 0..n {
                let vel = self.vel.eval(t, x[i]);
                let int = self.int.eval(t, x[i], x.iter().cloned());
                let tot = vel + int;
                let dens = if tot.is_sign_positive() {
                    if i < n - 1 {
                        f / (x[i + 1] - x[i])
                    } else {
                        X::zero()
                    }
                } else {
                    if i > 0 {
                        f / (x[i] - x[i - 1])
                    } else {
                        X::zero()
                    }
                };
                let m = self.mob.eval(t, dens);
                dx[i] = tot * m;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_velocity() {
        assert_eq!(ZeroVelocity.eval(0.0, 0.0), 0.0);
    }

    #[test]
    fn zero_interaction_potential() {
        let p = [1., 2., 3.];
        assert_eq!(ZeroInteraction.eval(0.0, 0.0, p), 0.0);
    }

    #[test]
    fn velocity() {
        let vel = Velocity(|t: f64, x: f64| t + x);
        assert_eq!(vel.eval(2., 3.), 5.);
    }

    #[test]
    fn sampled_interaction() {
        let int = SampledInteraction(|_t: f64, x: f64| x);
        let p = [0., 1., 2., 3.];
        assert_eq!(int.eval(0., 0., p), (0. + 1. + 2. + 3.) / (4. - 1.));
    }

    #[test]
    fn integrated_interaction() {
        let int = IntegratedInteraction(|_t: f64, x: f64| x * x / 2.);
        let p = [0., 1., 2., 3.];
        assert_eq!(int.eval(0., 0., p), (3. * 3. / 2. - 0.) / 3.);
    }
}
