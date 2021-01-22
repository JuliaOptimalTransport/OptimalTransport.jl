<a href="http://zsteve.phatcode.net/OptimalTransportDocs/">
<img src="images/optimaltransport_logo2.png" height="125"><br></a>

## Optimal Transport algorithms for Julia

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://zsteve.github.io/OptimalTransport.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://zsteve.github.io/OptimalTransport.jl/dev)
![CI](https://github.com/zsteve/OptimalTransport.jl/workflows/CI/badge.svg?branch=master)
[![Codecov](https://codecov.io/gh/zsteve/OptimalTransport.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/zsteve/OptimalTransport.jl)
[![Coveralls](https://coveralls.io/repos/github/zsteve/OptimalTransport.jl/badge.svg?branch=master)](https://coveralls.io/github/zsteve/OptimalTransport.jl?branch=master)

![example histogram](example.png)

This package provides some implementations of optimal transport algorithms, including the Sinkhorn scaling algorithm and its variants or extensions. 

## Supported algorithms

Currently, Earth Mover's Distance (solution to exact Monge-Kantorovich problem) is wrapped via PyCall using the excellent [POT (Python Optimal Transport)](https://pythonot.github.io/) library. The hope is that _gradually_, more and more functionality will be implemented in native Julia.

However, the following algorithms are offered in pure Julia:

* Sinkhorn algorithm (`sinkhorn`, `sinkhorn2`)
* Log-stabilized Sinkhorn algorithm (`sinkhorn_stabilized`)
* Epsilon-scaling stabilized Sinkhorn algorithm (`sinkhorn_stabilized_epsscaling`) 
* Unbalanced Sinkhorn algorithm (`sinkhorn_unbalanced`)
* Sinkhorn barycenter algorithm (`sinkhorn_barycenter`)
* Quadratically regularised optimal transport (`quadreg`)

See the documentation pages for further information. Most calling conventions are analogous to those in the Python Optimal Transport library.

## Documentation

Read the [documentation](http://zsteve.phatcode.net/OptimalTransportDocs/)

## Basic usage and examples

Click [here](http://zsteve.phatcode.net/OptimalTransportDocs/examples/examples.html) for a small collection of optimal transport examples using OptimalTransport.jl.

## Acknowledgements

Contributors include:

- Tim Matsumoto

- David Widmann

- Davi Barreira



## References

Peyré, G. and Cuturi, M., 2019. Computational optimal transport. Foundations and Trends® in Machine Learning, 11(5-6), pp.355-607.

Lorenz, D.A., Manns, P. and Meyer, C., 2019. Quadratically regularized optimal transport. Applied Mathematics & Optimization, pp.1-31.

Rémi Flamary and Nicolas Courty, POT Python Optimal Transport library, https://pythonot.github.io/, 2017

Chizat, L., Peyré, G., Schmitzer, B. and Vialard, F.X., 2016. Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

Schmitzer, B., 2019. Stabilized sparse scaling algorithms for entropy regularized transport problems. SIAM Journal on Scientific Computing, 41(3), pp.A1443-A1481.
