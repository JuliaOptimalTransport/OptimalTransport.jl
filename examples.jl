# Examples for OptimalTransport.jl
#
# Author: Stephen Zhang (syz@math.ubc.ca)

using OptimalTransport
using Distances

N = 10; M = 10
μ_spt = rand(N)
ν_spt = rand(M)

μ = ones(size(μ_spt, 1))
ν = ones(size(ν_spt, 1))

C = pairwise(Euclidean(), μ_spt', ν_spt').^2
ϵ = 0.01

using BenchmarkTools

@benchmark  _sinkhorn(μ, ν, C, ϵ)
@benchmark sinkhorn(μ, ν, C, ϵ)

N = 10; M = 20
μ_spt = rand(N)
ν_spt = rand(M)

μ = ones(size(μ_spt, 1))/N
ν = ones(size(ν_spt, 1))/M

C = pairwise(Euclidean(), μ_spt', ν_spt').^2
ϵ = 0.01
λ = 1
@benchmark _sinkhorn_unbalanced(μ, ν, C, ϵ, λ)
@benchmark sinkhorn_unbalanced(μ, ν, C, λ, λ, ϵ)

ϵ = 1e-4
γ = _sinkhorn_stabilized_epsscaling(μ, ν, C, ϵ)
γ_ = sinkhorn_stabilized_epsscaling(μ, ν, C, ϵ)

## try and make a pretty plot
using Seaborn

μ_spt = ν_spt = LinRange(-2, 2, 100)
C = pairwise(Euclidean(), μ_spt', ν_spt').^2
μ = exp.((-(μ_spt).^2)/0.5^2)
μ /= sum(μ)
ν = ν_spt.^2 .*exp.((-(ν_spt).^2)/0.5^2)
ν /= sum(ν)
γ = OptimalTransport.sinkhorn_stabilized(μ, ν, C, 1e-4, max_iter = 5000)
using Random, Distributions

F = DiscreteNonParametric(1:prod(size(γ)), reshape(γ, prod(size(γ))))
t = rand(F, 5*10^3)
μ_idx = t .% size(γ, 1)
ν_idx = t .÷ size(γ, 2)


Seaborn.figure()
Seaborn.jointplot(x = [μ_spt[i] for i in μ_idx], y = [ν_spt[i] for i in ν_idx], kind = "kde")
Seaborn.gcf()
Seaborn.savefig("example.png")
