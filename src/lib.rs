pub trait ExternalVelocity<T, X> {
    fn eval(&self, t: T, x: X) -> X;
}

pub trait InteractionPotential<T, X> {
    fn eval(&self, t: T, x: X) -> X;
}

pub trait Particles<X> {
    fn get<I>(i: I) -> X;
}

pub struct ZeroVelocity;
pub struct ZeroInteractionPotential;

impl<T, X> ExternalVelocity<T, X> for ZeroVelocity
where
    X: num_traits::Zero,
{
    fn eval(&self, _t: T, _x: X) -> X {
        X::zero()
    }
}

impl<T, X> InteractionPotential<T, X> for ZeroInteractionPotential
where
    X: num_traits::Zero,
{
    fn eval(&self, _t: T, _x: X) -> X {
        X::zero()
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
        assert_eq!(ZeroInteractionPotential.eval(0.0, 0.0), 0.0);
    }
}
