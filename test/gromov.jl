using OptimalTransport

using Distances
using PythonOT: PythonOT

using Random
using Test
using LinearAlgebra

const POT = PythonOT

Random.seed!(100)

M, N = 10, 10

μ = fill(1/M, M)
μ_spt = rand(M)
ν = fill(1/N, N)
ν_spt = rand(N)

Cμ = pairwise(SqEuclidean(), μ_spt)
Cν = pairwise(SqEuclidean(), ν_spt)

γ = OptimalTransport.entropic_gromov_wasserstein(μ, ν, Cμ, Cν, 0.01; check_convergence = 10)
γ_pot = PythonOT.entropic_gromov_wasserstein(μ, ν, Cμ, Cν, 0.01)

norm(γ .- γ_pot, 1)
norm(γ, 1)
norm(γ_pot, 1)
