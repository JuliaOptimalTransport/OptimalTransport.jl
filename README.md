# ot.jl
### Optimal transport algorithms for Julia.

Under construction: currently, Earth Mover's Distance (solution to exact problem) is wrapped via PyCall. However, the following algorithms are offered natively:

* Sinkhorn algorithm (`sinkhorn`, `sinkhorn2`, `sinkhorn_impl`)
* Unbalanced Sinkhorn algorithm (`sinkhorn_unbalanced`)

See `ot.jl` for further documentation. Most calling conventions are analogous to those in the Python OT library.

Example: 
```julia

N = 10; M = 10
μ_spt = rand(N)
ν_spt = rand(M)

μ = ones(size(μ_spt, 1))
ν = ones(size(ν_spt, 1))

C = pairwise(Euclidean(), μ_spt', ν_spt').^2
ϵ = 0.01

sinkhorn(μ, ν, C, ϵ)

N = 10; M = 20
μ_spt = rand(N)
ν_spt = rand(M)

μ = ones(size(μ_spt, 1))
ν = ones(size(ν_spt, 1))

C = pairwise(Euclidean(), μ_spt', ν_spt').^2
ϵ = 0.01
λ = 1
sinkhorn_unbalanced(μ, ν, C, λ, λ, ϵ)

``` 
