pub trait ExternalVelocity<T, X = T> {
    fn eval(&self, t: T, x: X) -> X;
}

pub trait Interaction<T, X = T> {
    fn eval<P>(&self, t: T, x: X, p: P) -> X
    where
        P: IntoIterator<Item = X>;
    // <P as IntoIterator>::IntoIter: ExactSizeIterator;
}

// pub trait InteractionPotential<T, P, X = T> {
//     fn eval(&self, t: T, x: X, p: P) -> X;
// }

// pub trait Particles<X> {
// fn get<I>(species: I, i: I) -> X;
// }

pub struct ZeroVelocity;
pub struct ZeroInteraction;

impl<T, X> ExternalVelocity<T, X> for ZeroVelocity
where
    X: num_traits::Zero,
{
    fn eval(&self, _t: T, _x: X) -> X {
        X::zero()
    }
}

impl<T, X> Interaction<T, X> for ZeroInteraction
where
    X: num_traits::Zero,
{
    fn eval<P>(&self, _t: T, _x: X, _p: P) -> X
    where
        P: IntoIterator<Item = X>,
        // <P as IntoIterator>::IntoIter: ExactSizeIterator,
    {
        X::zero()
    }
}

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
    fn eval(&self, t: T, x: X) -> X {
        self.0(t, x)
    }
}

impl<T, X, V> ExternalVelocity<T, X> for V
where
    V: Fn(T, X) -> X,
{
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
        // <P as IntoIterator>::IntoIter: ExactSizeIterator,
    {
        // -p.into_iter().map(|y| self.0(t, x - y)).sum::<X>()
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
        // <P as IntoIterator>::IntoIter: ExactSizeIterator,
    {
        let mut p = p.into_iter().peekable();
        // let len = p.len() - 1;
        // let f = X::from(len).unwrap().recip();
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
