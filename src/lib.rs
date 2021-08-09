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

pub struct ConservationLaw<T, X, Vel, Int, Mob> {
    pub vel: Vel,
    pub int: Int,
    pub mob: Mob,
    _t: std::marker::PhantomData<T>,
    _x: std::marker::PhantomData<X>,
}

#[cfg(feature = "ode_solver")]
pub use _ode_solver::*;

#[cfg(feature = "ode_solver")]
#[cfg_attr(docsrs, doc(cfg(feature = "ode_solver")))]
mod _ode_solver {
    use super::*;

    use ode_solvers::{SVector, System};

    impl<X, Vel, Int, Mob, const N: usize> System<SVector<X, N>>
        for ConservationLaw<f64, X, Vel, Int, Mob>
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
