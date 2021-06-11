# OptimalTransport.jl <a href='https://juliaoptimaltransport.github.io/OptimalTransport.jl/dev'><img src="docs/src/assets/logo.svg" align="right" height="138.5" /></a>

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaOptimalTransport.github.io/OptimalTransport.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaOptimalTransport.github.io/OptimalTransport.jl/dev)
[![CI](https://github.com/JuliaOptimalTransport/OptimalTransport.jl/workflows/CI/badge.svg?branch=master)](https://github.com/JuliaOptimalTransport/OptimalTransport.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![GPU tests](https://img.shields.io/buildkite/ec407153516cb9bc2a8b66105bc4a418d223ab9bba5f1cbe3a/master?label=GPU%20tests)](https://buildkite.com/julialang/optimaltransport-dot-jl/builds?branch=master)
[![DOI](https://zenodo.org/badge/253333137.svg)](https://zenodo.org/badge/latestdoi/253333137)
[![Codecov](https://codecov.io/gh/JuliaOptimalTransport/OptimalTransport.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaOptimalTransport/OptimalTransport.jl)
[![Coveralls](https://coveralls.io/repos/github/JuliaOptimalTransport/OptimalTransport.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaOptimalTransport/OptimalTransport.jl?branch=master)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

This package provides some [Julia](https://julialang.org/) implementations of algorithms for computational [optimal transport](https://optimaltransport.github.io/), including the Earth-Mover's (Wasserstein) distance, Sinkhorn algorithm for entropically regularized optimal transport as well as some variants or extensions.

Notably, OptimalTransport.jl provides GPU acceleration through [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl/) and [NNlibCUDA.jl](https://github.com/FluxML/NNlibCUDA.jl).

This package is inspired by the [Python Optimal Transport](https://optimaltransport.github.io/)
package.

## Example

```julia
using OptimalTransport
using Distances

# uniform histograms
μ = fill(1/250, 250)
ν = fill(1/200, 200)

# random cost matrix
C = pairwise(SqEuclidean(), rand(1, 250), rand(1, 200); dims=2)

# regularization parameter
ε = 0.01

# solve entropically regularized optimal transport problem
sinkhorn(μ, ν, C, ε)
```

Please see the documentation pages for further information.

## Related packages

- [PythonOT.jl](https://github.com/JuliaOptimalTransport/PythonOT.jl): Julia interface for the [Python Optimal Transport (POT) package](https://pythonot.github.io/).
- [StochasticOptimalTransport.jl](https://github.com/JuliaOptimalTransport/StochasticOptimalTransport.jl): Julia implementation of stochastic optimization algorithms for large-scale optimal transport.

## Contributing

Contributions are more than welcome! Please feel free to submit an issue or pull request in this repository.

## Acknowledgements

Contributors include:

- Tim Matsumoto (UBC)
- David Widmann (Uppsala)
- Davi Barreira (FGV)
- Stephen Zhang (UBC)

## References

- Peyré, G. and Cuturi, M., 2019. Computational optimal transport. Foundations and Trends® in Machine Learning, 11(5-6), pp.355-607.
- Lorenz, D.A., Manns, P. and Meyer, C., 2019. Quadratically regularized optimal transport. Applied Mathematics & Optimization, pp.1-31.
- Rémi Flamary and Nicolas Courty, POT Python Optimal Transport library, https://pythonot.github.io/, 2017
- Chizat, L., Peyré, G., Schmitzer, B. and Vialard, F.X., 2016. Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.
- Schmitzer, B., 2019. Stabilized sparse scaling algorithms for entropy regularized transport problems. SIAM Journal on Scientific Computing, 41(3), pp.A1443-A1481.
