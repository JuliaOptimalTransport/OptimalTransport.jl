using OptimalTransport

using Distances
using ForwardDiff
using LogExpFunctions
using PythonOT: PythonOT

using LinearAlgebra
using Random
using Test

const POT = PythonOT

Random.seed!(100)

@testset "sinkhorn_stabilized.jl" begin
    # size of source and target
    M = 250
    N = 200

    @testset "without epsilon scaling" begin
        # create two random histograms
        μ = normalize!(rand(M), 1)
        ν = normalize!(rand(N), 1)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute optimal transport map
        ε = 0.01
        γ = sinkhorn_stabilized(μ, ν, C, ε; maxiter=5_000)

        # compare with regular Sinkhorn algorithm
        γ_sinkhorn = sinkhorn(μ, ν, C, ε; maxiter=5_000)
        @test γ ≈ γ_sinkhorn

        # compare with POT
        γ_pot = POT.sinkhorn(μ, ν, C, ε; method="sinkhorn_stabilized", numItermax=5_000)
        @test γ ≈ γ_pot rtol = 1e-6
    end

    @testset "with epsilon scaling" begin
        # create two random histograms
        μ = normalize!(rand(M), 1)
        ν = normalize!(rand(N), 1)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute optimal transport map (Julia implementation + POT)
        ε = 0.01
        γ = sinkhorn_stabilized_epsscaling(μ, ν, C, ε; maxiter=5_000)

        # compare with regular Sinkhorn algorithm
        γ_sinkhorn = sinkhorn(μ, ν, C, ε; maxiter=5_000)
        @test γ ≈ γ_sinkhorn rtol = 1e-6

        # compare with stabilized Sinkhorn algorithm
        γ_sinkhorn_stabilized = sinkhorn_stabilized(μ, ν, C, ε; maxiter=5_000)
        @test γ ≈ γ_sinkhorn_stabilized rtol = 1e-6

        # compare with POT
        γ_pot = POT.sinkhorn(μ, ν, C, ε; method="sinkhorn_stabilized", numItermax=5_000)
        @test γ ≈ γ_pot rtol = 1e-6
    end

    @testset "AD" begin
        # random histograms with random cost matrix
        μ = normalize!(rand(M), 1)
        ν = normalize!(rand(N), 1)
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute gradients with respect to source and target marginals separately and
        # together
        ε = 0.01
        for f in (sinkhorn_stabilized, sinkhorn_stabilized_epsscaling)
            ForwardDiff.gradient(zeros(N)) do xs
                return dot(C, f(μ, softmax(xs), C, ε))
            end
            ForwardDiff.gradient(zeros(M)) do xs
                return dot(C, f(softmax(xs), ν, C, ε))
            end
            ForwardDiff.gradient(zeros(M + N)) do xs
                return dot(C, f(softmax(xs[1:M]), softmax(xs[(M + 1):end]), C, ε))
            end
        end
    end

    @testset "deprecations" begin
        # create two uniform histograms
        μ = fill(1 / M, M)
        ν = fill(1 / N, N)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # check `sinkhorn_stabilized`
        ε = 0.01
        γ = sinkhorn_stabilized(μ, ν, C, ε; atol=1e-6)
        @test (@test_deprecated sinkhorn_stabilized(μ, ν, C, ε; tol=1e-6)) == γ
        γ = sinkhorn_stabilized(μ, ν, C, ε)
        u, v = (@test_deprecated sinkhorn_stabilized(μ, ν, C, ε; return_duals=true))
        @test exp.(-(C .- u .- v') / ε) ≈ γ

        # check `sinkhorn_stabilized_epsscaling`
        γ = sinkhorn_stabilized_epsscaling(μ, ν, C, ε; scaling_steps=2)
        @test (@test_deprecated sinkhorn_stabilized_epsscaling(μ, ν, C, ε; k=2)) == γ
        γ = sinkhorn_stabilized_epsscaling(μ, ν, C, ε; scaling_factor=0.6)
        @test (@test_deprecated sinkhorn_stabilized_epsscaling(μ, ν, C, ε; lambda=0.6)) == γ
    end
end
