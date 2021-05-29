using OptimalTransport

using CUDA
using Distances

using Random
using Test

CUDA.allowscalar(false)

Random.seed!(100)

@testset "simple_gpu.jl" begin
    # ensure that a GPU is available
    if !CUDA.functional()
        @warn "skipped GPU tests: no GPU available"
    else
        @testset "sinkhorn" begin
            # source histogram
            m = 200
            μ = rand(Float32, m)
            μ ./= sum(μ)

            # target histogram
            n = 250
            ν = rand(Float32, n)
            ν ./= sum(ν)

            # random cost matrix
            C = pairwise(SqEuclidean(), randn(Float32, 1, m), randn(Float32, 1, n); dims=2)

            # compute transport plan and cost on the GPU
            ε = 0.01f0
            γ = sinkhorn(cu(μ), cu(ν), cu(C), ε)
            c = sinkhorn2(cu(μ), cu(ν), cu(C), ε)

            # compare with results on the CPU
            @test γ ≈ cu(sinkhorn(μ, ν, C, ε))
            @test c ≈ cu(sinkhorn2(μ, ν, C, ε))
        end

        @testset "sinkhorn_unbalanced" begin
            # source histogram
            m = 200
            μ = rand(Float32, m)
            μ ./= 1.5f0 * sum(μ)

            # target histogram
            n = 250
            ν = rand(Float32, n)
            ν ./= sum(ν)

            # random cost matrix
            C = pairwise(SqEuclidean(), randn(Float32, 1, m), randn(Float32, 1, n); dims=2)

            # compute transport plan and cost on the GPU
            ε = 0.01f0
            λ1 = 0.4f0
            λ2 = 0.6f0
            γ = sinkhorn_unbalanced(cu(μ), cu(ν), cu(C), λ1, λ2, ε)
            c = sinkhorn_unbalanced2(cu(μ), cu(ν), cu(C), λ1, λ2, ε)

            # compare with results on the CPU
            @test γ ≈ cu(sinkhorn_unbalanced(μ, ν, C, λ1, λ2, ε))
            @test c ≈ cu(sinkhorn_unbalanced2(μ, ν, C, λ1, λ2, ε))
        end
    end
end