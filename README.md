# Particle methods for 1D conservation laws

[![Crates.io](https://img.shields.io/crates/v/cons-laws)](https://crates.io/crates/cons-laws)
[![docs.rs](https://img.shields.io/docsrs/cons-laws)](https://docs.rs/cons-laws)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/FedericoStra/cons-laws/Rust)](https://github.com/FedericoStra/cons-laws/actions/workflows/rust.yml)
[![MIT license](https://img.shields.io/crates/l/cons-laws)](https://choosealicense.com/licenses/mit/)
![Lines of code](https://tokei.rs/b1/github/FedericoStra/cons-laws?category=code)

This crate implements the deterministic particle schemes described in the article
*Entropy solutions of non-local scalar conservation laws with congestion via deterministic particle method*, E. Radici, F. Stra (2021), [https://arxiv.org/abs/2107.10760](https://arxiv.org/abs/2107.10760).

You can cite the article as

```
@online{RadiciStra2021,
    title={Entropy solutions of non-local scalar conservation laws with congestion via deterministic particle method}, 
    author={Emanuela Radici and Federico Stra},
    year={2021},
    eprint={2107.10760},
    archivePrefix={arXiv},
    primaryClass={math.AP},
    url={https://arxiv.org/abs/2107.10760}
}
```

This is a reimplementation in Rust of the Julia package [ConservationLawsParticles.jl](https://github.com/FedericoStra/ConservationLawsParticles.jl).

The goal of this crate is to solve non-local conservation laws with congestion of the form

    ∂ₜρ + divₓ[ρ⋅v(ρ)⋅(V - ∂ₓW⋆ρ)] = 0

via deterministic particle schemes.
Plotting the trajectories of the particles produces images such as ![trajectories](doc/imgs/traffic_ode_solver.png)
