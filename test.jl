using ot

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

μ = ones(size(μ_spt, 1))
ν = ones(size(ν_spt, 1))

C = pairwise(Euclidean(), μ_spt', ν_spt').^2
ϵ = 0.01
λ = 1
@benchmark _sinkhorn_unbalanced(μ, ν, C, ϵ, λ)
@benchmark sinkhorn_unbalanced(μ, ν, C, λ, λ, ϵ)
