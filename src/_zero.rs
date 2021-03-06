use super::{ExternalVelocity, Interaction, Mobility};

pub struct ZeroVelocity;
pub struct ZeroInteraction;
pub struct OneMobility;

impl<T, X> ExternalVelocity<T, X> for ZeroVelocity
where
    X: num_traits::Zero,
{
    #[inline]
    fn eval(&self, _t: T, _x: X) -> X {
        X::zero()
    }
}

impl<T, X> Interaction<T, X> for ZeroInteraction
where
    X: num_traits::Zero,
{
    #[inline]
    fn eval<P>(&self, _t: T, _x: X, _p: P) -> X
    where
        P: IntoIterator<Item = X>,
    {
        X::zero()
    }
}

impl<T, X> Mobility<T, X> for OneMobility
where
    X: num_traits::One,
{
    #[inline]
    fn eval(&self, _t: T, _x: X) -> X {
        X::one()
    }
}
