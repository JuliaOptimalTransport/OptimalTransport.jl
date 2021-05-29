using OptimalTransport

using Distances
using Tulip
using Distributions
using HCubature

using LinearAlgebra
using Random

Random.seed!(7)

@testset "Multivariate Gaussians" begin
    @testset "translation with constant covariance" begin
        m = randn(100)
        τ = rand(100)
        Σ = Matrix(Hermitian(rand(100, 100) + 100I))
        μ = MvNormal(m, Σ)
        ν = MvNormal(m .+ τ, Σ)
        @test ot_cost(SqEuclidean(), μ, ν) ≈ norm(τ)^2

        x = rand(100, 10)
        T = ot_plan(SqEuclidean(), μ, ν)
        @test pdf(ν, mapslices(T, x; dims=1)) ≈ pdf(μ, x)
    end

    @testset "comparison to grid approximation" begin
        μ = MvNormal([0, 0], [1 0; 0 2])
        ν = MvNormal([10, 10], [2 0; 0 1])
        # Constructing circular grid approximation
        # Angular grid step
        θ = collect(0:0.2:(2π))
        θx = cos.(θ)
        θy = sin.(θ)
        # Radius grid step
        δ = collect(0:0.2:1)
        μsupp = [0.0 0.0]
        νsupp = [10.0 10.0]
        for i in δ[2:end]
            a = [θx .* i θy .* i * 2]
            b = [θx .* i * 2 θy .* i] .+ [10 10]
            μsupp = vcat(μsupp, a)
            νsupp = vcat(νsupp, b)
        end

        # Create discretized distribution
        μprobs = pdf(μ, μsupp')
        μprobs = μprobs ./ sum(μprobs)
        νprobs = pdf(ν, νsupp')
        νprobs = νprobs ./ sum(νprobs)
        C = pairwise(SqEuclidean(), μsupp', νsupp')
        @test isapprox(
            emd2(μprobs, νprobs, C, Tulip.Optimizer()),
            ot_cost(SqEuclidean(), μ, ν);
            rtol=1e-3,
        )

        # Use hcubature integration to perform ``\\int c(x,T(x)) d\\mu``
        T = ot_plan(SqEuclidean(), μ, ν)
        f(x) = sqeuclidean(x, T(x)) * pdf(μ, x)
        @test isapprox(
            hcubature(f, [-10, -10], [10, 10])[1], ot_cost(SqEuclidean(), μ, ν); rtol=1e-3
        )
    end
end
