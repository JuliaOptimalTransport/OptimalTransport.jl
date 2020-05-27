using OptimalTransport
using CUDA
using Distances

using LinearAlgebra
using Random
using Test

Random.seed!(100)

@testset "entropically regularized transport" begin
    M = 250
    N = 200

    @testset "example" begin
        # create two uniform histograms
        μ = fill(1/M, M)
        ν = fill(1/N, N)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims = 2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.01
        γ = sinkhorn(μ, ν, C, eps)
        γ_pot = OptimalTransport.pot_sinkhorn(μ, ν, C, eps)
        @test norm(γ - γ_pot, Inf) < 1e-9

        # compute optimal transport cost (Julia implementation + POT)
        c = sinkhorn2(μ, ν, C, eps)
        c_pot = OptimalTransport.pot_sinkhorn2(μ, ν, C, eps)
        @test c ≈ c_pot atol=1e-9
    end

    # different element type
    @testset "Float32" begin
        # create two uniform histograms
        μ = fill(Float32(1/M), M)
        ν = fill(Float32(1/N), N)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(Float32, 1, M), rand(Float32, 1, N); dims = 2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.01f0
        γ = sinkhorn(μ, ν, C, eps)
        @test eltype(γ) === Float32

        γ_pot = OptimalTransport.pot_sinkhorn(μ, ν, C, eps)
        @test eltype(γ_pot) === Float64 # POT does not respect input type
        @test norm(γ - γ_pot, Inf) < Base.eps(Float32)

        # compute optimal transport cost (Julia implementation + POT)
        c = sinkhorn2(μ, ν, C, eps)
        @test c isa Float32

        c_pot = OptimalTransport.pot_sinkhorn2(μ, ν, C, eps)
        @test c_pot isa Float64 # POT does not respect input types
        @test c ≈ c_pot atol=Base.eps(Float32)
    end

    # computation on the GPU
    if CUDA.functional()
        @testset "CUDA" begin
            # create two uniform histograms
            μ = CUDA.fill(Float32(1/M), M)
            ν = CUDA.fill(Float32(1/N), N)

            # create random cost matrix
            C = abs2.(CUDA.rand(M) .- CUDA.rand(1, N))

            # compute optimal transport map
            eps = 0.01f0
            γ = sinkhorn(μ, ν, C, eps)
            @test γ isa CuArray{Float32}

            # compute optimal transport cost
            c = sinkhorn2(μ, ν, C, eps)
            @test c isa Float32
        end
    end
end
