use super::{ExternalVelocity, Interaction};

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
    {
        X::zero()
    }
}
