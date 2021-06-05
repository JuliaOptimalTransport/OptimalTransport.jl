using OptimalTransport

using CUDA
using Distances

using LinearAlgebra
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
            μ = normalize!(rand(Float32, m), 1)
            cu_μ = cu(μ)

            # target histogram
            n = 250
            ν = normalize!(rand(Float32, n), 1)
            cu_ν = cu(ν)

            # random cost matrix
            C = pairwise(SqEuclidean(), randn(Float32, 1, m), randn(Float32, 1, n); dims=2)
            cu_C = cu(C)

            # compute transport plan and cost on the GPU
            ε = 0.01f0
            γ = sinkhorn(cu_μ, cu_ν, cu_C, ε)
            @test γ isa CuArray{Float32,2}
            c = sinkhorn2(cu_μ, cu_ν, cu_C, ε)
            @test c isa Float32

            # compare with results on the CPU
            γ_cpu = sinkhorn(μ, ν, C, ε)
            @test convert(Array, γ) ≈ γ_cpu
            @test c ≈ sinkhorn2(μ, ν, C, ε)

            # batches of histograms
            d = 10
            for (size2_μ, size2_ν) in
                (((), (d,)), ((1,), (d,)), ((d,), ()), ((d,), (1,)), ((d,), (d,)))
                # generate uniform histograms
                μ_batch = repeat(cu_μ, 1, size2_μ...)
                ν_batch = repeat(cu_ν, 1, size2_ν...)

                # compute optimal transport plan and check that it is consistent with the
                # plan for individual histograms
                γ_all = sinkhorn(μ_batch, ν_batch, cu_C, ε)
                @test γ_all isa CuArray{Float32,3}
                @test size(γ_all) == (m, n, d)
                @test all(γi ≈ γ_cpu for γi in eachslice(convert(Array, γ_all); dims=3))

                # compute optimal transport cost and check that it is consistent with the
                # cost for individual histograms
                c_all = sinkhorn2(μ_batch, ν_batch, cu_C, ε)
                @test c_all isa CuArray{Float32,1}
                @test size(c_all) == (d,)
                @test all(ci ≈ c for ci in convert(Array, c_all))
            end
        end

        @testset "sinkhorn_unbalanced" begin
            # source histogram
            m = 200
            μ = normalize!(rand(Float32, m), 1)
            μ .*= 1.5f0

            # target histogram
            n = 250
            ν = normalize!(rand(Float32, n), 1)

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
