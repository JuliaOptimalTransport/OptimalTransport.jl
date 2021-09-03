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

@testset "dual.jl" begin
    M = 250
    N = 200

    # create two random histograms
    μ = normalize!(rand(M), 1)
    ν = normalize!(rand(N), 1)

    # create random cost matrix
    C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)
    ε = 0.01
    K = exp.(-C/ε)

    @testset "semidual_grad" begin
        ∇_ad = ForwardDiff.gradient(zeros(size(ν))) do xs
            Dual.ot_entropic_semidual(μ, xs, ε, K; stabilized = false)
        end

        ∇ = Dual.ot_entropic_semidual_grad(μ, zeros(size(ν)), ε, K)

        @test ∇ ≈ ∇_ad 
    end

    @testset "dual_grad" begin
        ∇_ad = ForwardDiff.gradient(zeros(size(ν, 1) + size(μ, 1))) do xs
            Dual.ot_entropic_dual(xs[1:M], xs[M+1:end], ε, K)
        end

        ∇ = vcat(Dual.ot_entropic_dual_grad(zeros(size(μ)), zeros(size(ν)), ε, K)...)

        @test ∇ ≈ ∇_ad
    end

    @testset "batch" begin
        L = 10
        μ = rand(M, L)
        μ = μ ./ sum(μ; dims = 1)
        u = rand(size(μ)...)
        ν = rand(N, L)
        ν = ν ./ sum(ν; dims = 1)
        v = rand(size(ν)...)
        @testset "dual" begin
            @test Dual.ot_entropic_dual(u, v, ε, K) == [Dual.ot_entropic_dual(x, y, ε, K) for (x, y) in zip(eachcol(u), eachcol(v))]

            ∇_u, ∇_v = Dual.ot_entropic_dual_grad(u, v, ε, K)
            grad_pairwise = [Dual.ot_entropic_dual_grad(x, y, ε, K) for (x, y) in zip(eachcol(u), eachcol(v))]
            @test (∇_u ≈ hcat([g[1] for g in grad_pairwise]...)) && (∇_v ≈ hcat([g[2] for g in grad_pairwise]...))

            @test Dual.getprimal_ot_entropic_dual(u, v, ε, K) ≈ cat([Dual.getprimal_ot_entropic_dual(x, y, ε, K) for (x, y) in zip(eachcol(u), eachcol(v))]...; dims = 3)
        end
        
        @testset "semidual" begin
            @test Dual.ot_entropic_semidual(μ, v, ε, K) == [Dual.ot_entropic_semidual(x, y, ε, K) for (x, y) in zip(eachcol(μ), eachcol(v))]
            @test Dual.ot_entropic_semidual_grad(μ, v, ε, K) ≈ hcat([Dual.ot_entropic_semidual_grad(x, y, ε, K) for (x, y) in zip(eachcol(μ), eachcol(v))]...)

            @test Dual.getprimal_ot_entropic_semidual(μ, v, ε, K) ≈ hcat([Dual.getprimal_ot_entropic_semidual(x, y, ε, K) for (x, y) in zip(eachcol(μ), eachcol(v))]...)
        end
    end
end
