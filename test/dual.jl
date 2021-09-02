using OptimalTransport
import OptimalTransport.Dual: Dual

using ForwardDiff

using Distances
using PythonOT: PythonOT

using LinearAlgebra
using Random
using Test

const POT = PythonOT

Random.seed!(100)

@testset "dual" begin
    M = 250
    N = 200

    # create two random histograms
    μ = normalize!(rand(M), 1)
    ν = normalize!(rand(N), 1)

    # create random cost matrix
    C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)
    ε = 0.01
    K = exp.(-C/ε)

    ∇_ad = ForwardDiff.gradient(zeros(size(ν))) do xs
        Dual.ot_entropic_semidual(μ, xs, ε, K; stabilized = false)
    end

    ∇ = Dual.ot_entropic_semidual_grad(μ, zeros(size(ν)), ε, K)

    @test ∇ ≈ ∇_ad 
end
