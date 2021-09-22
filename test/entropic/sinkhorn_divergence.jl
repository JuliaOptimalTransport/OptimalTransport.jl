using OptimalTransport

using Distances
using ForwardDiff
using ReverseDiff
using LogExpFunctions
using PythonOT: PythonOT
using LinearAlgebra
using Random
using Test
using Compat

const POT = PythonOT

Random.seed!(100)

@testset "sinkhorn_divergence.jl" begin
    @testset "fixed_support" begin
        # size of problem
        N = 250
        # number of target measures 
        x = range(-1, 1; length=N)
        C = pairwise(SqEuclidean(), x)
        f(x; μ, σ) = exp(-((x - μ) / σ)^2)
        # regularization parameter
        ε = 0.05
        @testset "basic" begin
            μ = normalize!(f.(x; μ=0, σ=0.5), 1)
            M = 100

            ν_all = [normalize!(f.(x; μ=y, σ=0.5), 1) for y in range(-1, 1; length=M)]

            for reg in (true, false)
                loss = map(ν -> sinkhorn_divergence(μ, ν, C, ε; regularization=reg), ν_all)
                loss_ = map(
                    ν ->
                        sinkhorn2(μ, ν, C, ε; regularization=reg) -
                        (
                            sinkhorn2(μ, μ, C, ε; regularization=reg) +
                            sinkhorn2(ν, ν, C, ε; regularization=reg)
                        ) / 2,
                    ν_all,
                )

                @test loss ≈ loss_ rtol = 1e-6
                @test all(loss .≥ 0)
                @test sinkhorn_divergence(μ, μ, C, ε) ≈ 0 atol = 1e-9
            end
        end
        @testset "batch" begin
            M = 10
            μ = hcat([normalize!(f.(x; μ=randn(), σ=0.5), 1) for _ in 1:M]...)
            ν = hcat([normalize!(f.(x; μ=randn(), σ=0.5), 1) for _ in 1:M]...)
            for reg in (true, false)
                loss_batch = sinkhorn_divergence(μ, ν, C, ε; regularization=reg)
                @compat @test loss_batch ≈ [
                    sinkhorn_divergence(x, y, C, ε; regularization=reg) for
                    (x, y) in zip(eachcol(μ), eachcol(ν))
                ]
                loss_batch_μ = sinkhorn_divergence(μ, ν[:, 1], C, ε; regularization=reg)
                @compat @test loss_batch_μ ≈ [
                    sinkhorn_divergence(x, ν[:, 1], C, ε; regularization=reg) for
                    x in eachcol(μ)
                ]
                loss_batch_ν = sinkhorn_divergence(μ[:, 1], ν, C, ε; regularization=reg)
                @compat @test loss_batch_ν ≈ [
                    sinkhorn_divergence(μ[:, 1], y, C, ε; regularization=reg) for
                    y in eachcol(ν)
                ]
            end
        end
        @testset "AD" begin
            ε = 0.05
            μ = normalize!(f.(x; μ=-0.5, σ=0.5), 1)
            ν = normalize!(f.(x; μ=0.5, σ=0.5), 1)
            for Diff in [ForwardDiff, ReverseDiff]
                for reg in (true, false)
                    ∇ = Diff.gradient(log.(ν)) do xs
                        sinkhorn_divergence(μ, softmax(xs), C, ε; regularization=reg)
                    end
                    @test size(∇) == size(ν)
                    ∇ = Diff.gradient(log.(μ)) do xs
                        sinkhorn_divergence(μ, softmax(xs), C, ε; regularization=reg)
                    end
                    @test norm(∇, Inf) ≈ 0 atol = 1e-9 # Sinkhorn divergence has minimum at SD(μ, μ)
                end
            end
        end
    end
    @testset "empirical" begin
        N = 50
        M = 64
        d = 2
        μ_spt = randn(N, d)
        ν_spt = 1.5randn(M, d)
        μ = fill(1 / N, N)
        ν = fill(1 / M, M)
        Cμν = pairwise(SqEuclidean(), μ_spt', ν_spt'; dims=2)
        Cμ = pairwise(SqEuclidean(), μ_spt'; dims=2)
        Cν = pairwise(SqEuclidean(), ν_spt'; dims=2)
        ε = 1.0

        @testset "basic" begin
            for reg in (true, false)
                @test sinkhorn_divergence(μ, ν, Cμν, Cμ, Cν, ε; regularization=reg) ≥ 0
                @test sinkhorn_divergence(μ, μ, Cμ, Cμ, Cμ, ε; regularization=reg) ≈ 0 atol =
                    1e-6
            end
        end

        @testset "AD" begin
            for Diff in [ForwardDiff, ReverseDiff]
                for reg in (true, false)
                    ∇ = Diff.gradient(ν_spt) do xs
                        Cμν = pairwise(SqEuclidean(), μ_spt', xs'; dims=2)
                        Cν = pairwise(SqEuclidean(), xs'; dims=2)
                        sinkhorn_divergence(μ, ν, Cμν, Cμ, Cν, ε)
                    end
                    @test size(∇) == size(ν_spt)
                    ∇ = Diff.gradient(μ_spt) do xs
                        Cμν = pairwise(SqEuclidean(), μ_spt', xs'; dims=2)
                        Cν = pairwise(SqEuclidean(), xs'; dims=2)
                        sinkhorn_divergence(μ, μ, Cμν, Cμ, Cν, ε)
                    end
                    @test norm(∇, Inf) ≈ 0 atol = 1e-6
                end
            end
        end
    end
end
