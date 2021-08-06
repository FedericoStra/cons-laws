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
//! This is a reimplementation in Rust of the Julia package [ConservationLawsParticles.jl](https://github.com/FedericoStra/ConservationLawsParticles.jl).
//!
//! $1+2$
//!
//! $$
//! \frac{dy}{dt}=f(t, y)
//! $$

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

mod _zero;
pub use _zero::{ZeroInteraction, ZeroVelocity};

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
