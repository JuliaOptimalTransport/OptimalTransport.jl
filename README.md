<a href="https://zsteve.github.io/OptimalTransport.jl/dev">
<img src="images/optimaltransport_logo.png" height="125"><br></a>

## Optimal transport algorithms for Julia

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://zsteve.github.io/OptimalTransport.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://zsteve.github.io/OptimalTransport.jl/dev)
[![CI](https://github.com/zsteve/OptimalTransport.jl/workflows/CI/badge.svg?branch=master)](https://github.com/zsteve/OptimalTransport.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![Codecov](https://codecov.io/gh/zsteve/OptimalTransport.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/zsteve/OptimalTransport.jl)
[![Coveralls](https://coveralls.io/repos/github/zsteve/OptimalTransport.jl/badge.svg?branch=master)](https://coveralls.io/github/zsteve/OptimalTransport.jl?branch=master)

This package provides some [Julia](https://julialang.org/) implementations of algorithms for computational [optimal transport](https://optimaltransport.github.io/), including the Earth-Mover's (Wasserstein) distance, Sinkhorn scaling algorithm for entropy-regularised transport as well as some variants or extensions. 

## Overview of supported functionality

* Monge-Kantorovich (Earth-Mover's) distance (`emd`, `emd2`)
* Sinkhorn algorithm for entropy-regularised optimal transport (`sinkhorn`, `sinkhorn2`)
* Log-stabilized Sinkhorn algorithm (`sinkhorn_stabilized`)
* Epsilon-scaling stabilized Sinkhorn algorithm (`sinkhorn_stabilized_epsscaling`) 
* Unbalanced Sinkhorn algorithm (`sinkhorn_unbalanced`)
* Entropy-regularised barycenters (Sinkhorn barycenters) (`sinkhorn_barycenter`)
* Quadratically regularised optimal transport via semismooth Newton scheme [Lorenz, 2019] (`quadreg`) 

See the documentation pages linked below for further information. Most calling conventions are analogous to those in the [Python Optimal Transport](https://optimaltransport.github.io/) library, which formed the inspiration for this library.

## Documentation

 - [Stable](https://zsteve.github.io/OptimalTransport.jl/stable)
 - [Dev](https://zsteve.github.io/OptimalTransport.jl/dev)

## Contributing

Contributions are more than welcome! Please feel free to submit an issue or pull request in this repository.

## Acknowledgements

Contributors include:

- Tim Matsumoto (UBC)
- David Widmann (Uppsala)
- Davi Barreira (FGV)

## References

- Peyré, G. and Cuturi, M., 2019. Computational optimal transport. Foundations and Trends® in Machine Learning, 11(5-6), pp.355-607.
- Lorenz, D.A., Manns, P. and Meyer, C., 2019. Quadratically regularized optimal transport. Applied Mathematics & Optimization, pp.1-31.
- Rémi Flamary and Nicolas Courty, POT Python Optimal Transport library, https://pythonot.github.io/, 2017
- Chizat, L., Peyré, G., Schmitzer, B. and Vialard, F.X., 2016. Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.
- Schmitzer, B., 2019. Stabilized sparse scaling algorithms for entropy regularized transport problems. SIAM Journal on Scientific Computing, 41(3), pp.A1443-A1481.
