# OptimalTransport.jl
---
Optimal transport algorithms for Julia.

[![Build Status](https://travis-ci.com/zsteve/OptimalTransport.jl.svg?branch=master)](https://travis-ci.com/zsteve/OptimalTransport.jl.svg?branch=master)

![example histogram](example.png)

Read the [documentation](http://zsteve.phatcode.net/OptimalTransportDocs/)!

Currently, Earth Mover's Distance (solution to exact problem) is wrapped via PyCall using the excellent [POT (Python Optimal Transport)](https://github.com/PythonOT/POT) library. The hope is that _gradually_, more and more functionality will be implemented in native Julia.

However, the following algorithms are offered natively:

* Sinkhorn algorithm (`sinkhorn`, `sinkhorn2`, `sinkhorn_impl`)
* Log-stabilized Sinkhorn algorithm (`sinkhorn_stabilized`)
* Epsilon-scaling stabilized Sinkhorn algorithm (`sinkhorn_stabilized_epsscaling`) [warning: not fully functional yet]
* Unbalanced Sinkhorn algorithm (`sinkhorn_unbalanced`)
* now registered on the general registry!

See the documentation pages or read `OptimalTransport.jl` for further documentation. Most calling conventions are analogous to those in the Python OT library.

## Examples

```julia
using OptimalTransport
using Distances
using LinearAlgebra

## Entropically regularised transport

N = 200; M = 200
μ_spt = rand(N)
ν_spt = rand(M)

μ = normalize!(ones(size(μ_spt, 1)), 1)
ν = normalize!(ones(size(ν_spt, 1)), 1)

C = pairwise(Euclidean(), μ_spt', ν_spt').^2
ϵ = 0.01

γ = sinkhorn(μ, ν, C, ϵ)
γ_ = _sinkhorn(μ, ν, C, ϵ)

norm(γ - γ_, Inf) # Check that we get the same result as POT

## Unbalanced transport

N = 200; M = 200
μ_spt = rand(N)
ν_spt = rand(M)

μ = normalize!(ones(size(μ_spt, 1)), 1)
ν = normalize!(ones(size(ν_spt, 1)), 1)

C = pairwise(Euclidean(), μ_spt', ν_spt').^2
ϵ = 0.01
λ = 1.0

γ_ = _sinkhorn_unbalanced(μ, ν, C, ϵ, λ)
γ = sinkhorn_unbalanced(μ, ν, C, λ, λ, ϵ)

norm(γ - γ_, Inf) # Check that we get the same result as POT
```

Code to generate example plot:
```julia
## Example plots 
using Seaborn

function sample_joint(γ, N)
    F = DiscreteNonParametric(1:prod(size(γ)), reshape(γ, prod(size(γ))))
    t = rand(F, N)
    μ_idx = ((t .- 1) .% size(γ, 1)) .+ 1
    ν_idx = ((t .- 1) .÷ size(γ, 2)) .+ 1
    return μ_idx, ν_idx
end

μ_spt = ν_spt = LinRange(-2, 2, 100)
C = pairwise(Euclidean(), μ_spt', ν_spt').^2
μ = exp.((-(μ_spt).^2)/0.5^2)
μ /= sum(μ)
ν = ν_spt.^2 .*exp.((-(ν_spt).^2)/0.5^2)
ν /= sum(ν)
γ = OptimalTransport.sinkhorn(μ, ν, C, 0.01)
using Random, Distributions
μ_idx, ν_idx = sample_joint(γ, 10000)

Seaborn.jointplot(x = [μ_spt[i] for i in μ_idx], y = [ν_spt[i] for i in ν_idx], kind = "hex", marginal_kws = Dict("bins" => μ_spt))
Seaborn.savefig("example.png")
```

## References

Peyré, G. and Cuturi, M., 2019. Computational optimal transport. Foundations and Trends® in Machine Learning, 11(5-6), pp.355-607.

