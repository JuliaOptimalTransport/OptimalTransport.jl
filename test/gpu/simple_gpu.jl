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

            # batches of histograms
            d = 10
            for (size2_μ, size2_ν) in
                (((), (d,)), ((1,), (d,)), ((d,), ()), ((d,), (1,)), ((d,), (d,)))
                # generate uniform histograms
                μ_batch = repeat(μ, 1, size2_μ...)
                ν_batch = repeat(ν, 1, size2_ν...)

                # compute optimal transport plan and check that it is consistent with the
                # plan for individual histograms
                γ_all = sinkhorn(μ_batch, ν_batch, C, ε)
                @test size(γ_all) == (m, n, d)
                @test all(isapprox.(γ_all, γ))

                # compute optimal transport cost and check that it is consistent with the
                # cost for individual histograms
                c_all = sinkhorn2(μ, ν, C, eps; maxiter=5_000, rtol=1e-9)
                @test size(c_all) == (d,)
                @test all(isapprox.(c_all, c))
            end
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
