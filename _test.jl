using OptimalTransport

using Distances
using ForwardDiff
using LogExpFunctions
using PythonOT: PythonOT

using LinearAlgebra
using Random
using Test
using StatsBase


# size of source and target
M = 250
N = 200

# create two random histograms
μ = normalize!(rand(M), 1)
ν = normalize!(rand(N), 1)

# create random cost matrix
C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

# regularization parameter
ε = 0.025*mean(C)
s = OptimalTransport.build_solver(μ, ν, C, ε, OptimalTransport.SinkhornDual(true, true); maxiter = 250)
OptimalTransport.solve!(s)
γ = OptimalTransport.plan(s)

γ0 = sinkhorn(μ, ν, C, ε)

γ0 ./ γ

@test γ ≈ γ0 rtol = 1e-3
@test dot(γ, C) ≈ dot(γ0, C) rtol = 1e-4

