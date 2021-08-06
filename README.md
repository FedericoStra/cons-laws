# Particle methods for 1D conservation laws

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://docs.rs/cons-laws)
[![Build Status](https://github.com/FedericoStra/cons-laws/actions/workflows/rust.yml/badge.svg)](https://github.com/FedericoStra/cons-laws/actions/workflows/rust.yml)

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
