# OptimalTransport.jl
### Optimal transport algorithms for Julia.

![example histogram](example.png)

Under construction!

Currently, Earth Mover's Distance (solution to exact problem) is wrapped via PyCall using the excellent [POT (Python Optimal Transport)](https://github.com/PythonOT/POT) library. The hope is that _gradually_, more and more functionality will be implemented in native Julia.

However, the following algorithms are offered natively:

* Sinkhorn algorithm (`sinkhorn`, `sinkhorn2`, `sinkhorn_impl`)
* Log-stabilized Sinkhorn algorithm (`sinkhorn_stabilized`)
* Epsilon-scaling stabilized Sinkhorn algorithm (`sinkhorn_stabilized_epsscaling`) [warning: not fully functional yet]
* Unbalanced Sinkhorn algorithm (`sinkhorn_unbalanced`)
* now registered on the general registry!

See `ot.jl` for further documentation. Most calling conventions are analogous to those in the Python OT library.

## Examples

```julia

N = 10; M = 10
μ_spt = rand(N)
ν_spt = rand(M)

μ = ones(size(μ_spt, 1))
ν = ones(size(ν_spt, 1))

C = pairwise(Euclidean(), μ_spt', ν_spt').^2
ϵ = 0.01

sinkhorn(μ, ν, C, ϵ)
sinkhorn_stabilized(μ, ν, C, ϵ)
sinkhorn_stabilized_epsscaling(μ, ν, C, ϵ)

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

Code to generate example plot:
```julia
## try and make a pretty plot
using Seaborn, Random, Distributions

μ_spt = ν_spt = LinRange(-2, 2, 100)
C = pairwise(Euclidean(), μ_spt', ν_spt').^2
μ = exp.((-(μ_spt).^2)/0.5^2)
μ /= sum(μ)
ν = ν_spt.^2 .*exp.((-(ν_spt).^2)/0.5^2)
ν /= sum(ν)
γ = OptimalTransport.sinkhorn_stabilized(μ, ν, C, 1e-4, max_iter = 5000)

F = DiscreteNonParametric(1:prod(size(γ)), reshape(γ, prod(size(γ))))
t = rand(F, 5*10^3)
μ_idx = t .% size(γ, 1)
ν_idx = t .÷ size(γ, 2)


Seaborn.figure()
Seaborn.jointplot(x = [μ_spt[i] for i in μ_idx], y = [ν_spt[i] for i in ν_idx], kind = "kde")
Seaborn.gcf()
Seaborn.savefig("example.png")
```

## References
TODO. Check the POT documentation for now.
