# Examples for OptimalTransport.jl
#
# Author: Stephen Zhang (syz@math.ubc.ca)

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

γ = OptimalTransport.sinkhorn(μ, ν, C, ϵ)
γ_ = OptimalTransport.pot_sinkhorn(μ, ν, C, ϵ)

norm(γ - γ_, Inf) # Check that we get the same result as POT

## Unbalanced transport

N = 100; M = 200
μ_spt = rand(N)
ν_spt = rand(M)

μ = normalize!(ones(size(μ_spt, 1)), 1)
ν = normalize!(ones(size(ν_spt, 1)), 1)

C = pairwise(Euclidean(), μ_spt', ν_spt').^2
ϵ = 0.05
λ = 1.0

γ_ = pot_sinkhorn_unbalanced(μ, ν, C, ϵ, λ)
γ = sinkhorn_unbalanced(μ, ν, C, λ, λ, ϵ, verbose = true)

norm(γ - γ_, Inf) # Check that we get the same result as POT

## Stabilized Sinkhorn
ϵ = 0.005
γ =  OptimalTransport.sinkhorn_stabilized(μ, ν, C, ϵ, max_iter = 5000)
γ_ = OptimalTransport.pot_sinkhorn(μ, ν, C, ϵ, method = "sinkhorn_stabilized", max_iter = 5000)

norm(γ - γ_, Inf) # Check that we get the same result as POT

## Stabilized Sinkhorn eps-scaling

γ =  OptimalTransport.sinkhorn_stabilized_epsscaling(μ, ν, C, ϵ, max_iter = 5000)
γ_ = OptimalTransport.pot_sinkhorn(μ, ν, C, ϵ, method = "sinkhorn_epsilon_scaling", max_iter = 5000)

norm(γ - γ_, Inf) # Check that we get the same result as POT

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
Seaborn.axes("equal")

Seaborn.savefig("example.png")
