using OptimalTransport

using Distances
using ForwardDiff
using LogExpFunctions
using PythonOT: PythonOT

using LinearAlgebra
using Random
using Test


# size of source and target
M = 250
N = 200

# create two random histograms
μ = normalize!(rand(M), 1)
ν = normalize!(rand(N), 1)

# create random cost matrix
C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

# regularization parameter
ε = 0.01

s = OptimalTransport.build_solver(μ, ν, C, ε, OptimalTransport.SinkhornDual())
OptimalTransport.solve!(s)
γ = OptimalTransport.plan(s)
