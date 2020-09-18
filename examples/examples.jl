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

μ = fill(1/N, N)
ν = fill(1/M, M) 
C = pairwise(SqEuclidean(), μ_spt', ν_spt')
ϵ = 0.01

γ = sinkhorn(μ, ν, C, ϵ)
γ_ = OptimalTransport.pot_sinkhorn(μ, ν, C, ϵ)

norm(γ - γ_, Inf) # Check that we get the same result as POT

## Quadratically regularised transport.
# There is not an implementation of this method in the Python OT library, 
# so we don't have an output to compare to.

γ = OptimalTransport.quadreg(μ, ν, C, ϵ)

## Stabilized Sinkhorn
ϵ = 0.005
γ =  sinkhorn_stabilized(μ, ν, C, ϵ, max_iter = 5000)
γ_ = pot_sinkhorn(μ, ν, C, ϵ, method = "sinkhorn_stabilized", max_iter = 5000)

norm(γ - γ_, Inf) # Check that we get the same result as POT

## Stabilized Sinkhorn eps-scaling

γ =  sinkhorn_stabilized_epsscaling(μ, ν, C, ϵ, max_iter = 5000)
γ_ = pot_sinkhorn(μ, ν, C, ϵ, method = "sinkhorn_epsilon_scaling", max_iter = 5000)

norm(γ - γ_, Inf) # Check that we get the same result as POT

## Unbalanced transport

N = 100; M = 200
μ_spt = rand(N)
ν_spt = rand(M)

μ = fill(1/N, N)
ν = fill(1/N, M)
C = pairwise(SqEuclidean(), μ_spt', ν_spt')
ϵ = 0.01
λ = 1.0

γ_ = pot_sinkhorn_unbalanced(μ, ν, C, ϵ, λ)
γ = sinkhorn_unbalanced(μ, ν, C, λ, λ, ϵ)

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


γ_quad = Matrix(OptimalTransport.quadreg(μ, ν, C, 5, maxiter = 500))
μ_idx, ν_idx = sample_joint(γ_quad/sum(γ_quad), 10000)

Seaborn.jointplot(x = [μ_spt[i] for i in μ_idx], y = [ν_spt[i] for i in ν_idx], kind = "hex", marginal_kws = Dict("bins" => μ_spt))
Seaborn.axes("equal")
Seaborn.savefig("example_quad.png")

## Sinkhorn barycenters

using PyPlot
spt = LinRange(-1, 1, 100)
f(x, σ) = exp.(-x.^2/σ^2)
normalize(x) = x./sum(x)
mu_all = hcat([normalize(f(spt .- z, 0.1)) for z in [-0.5, 0.5]]...)'
C_all = [pairwise(SqEuclidean(), spt', spt') for i = 1:size(mu_all, 1)]

a = sinkhorn_barycenter(mu_all, C_all, 0.05, [0.75, 0.25]; max_iter = 1000);
figure()
for i = 1:size(mu_all, 1)
    plot(spt, mu_all[i, :])
end
plot(spt, a)
