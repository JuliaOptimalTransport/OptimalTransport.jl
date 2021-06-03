using Distributions: DiscreteNonParametric
using OptimalTransport
using Distributions
using Distances
using Random
using StatsBase


Random.seed!(7)

@testset "costmatrix.jl" begin
    @testset "Creating cost matrices from vectors" begin
        N = 15
        M = 20
        μ = FiniteDiscreteMeasure(rand(N), rand(N))
        ν = FiniteDiscreteMeasure(rand(M), rand(M))
        c(x,y) = sum((x-y).^2)
        C1 = cost_matrix(SqEuclidean(), μ, ν)
        C2 = cost_matrix(sqeuclidean, μ, ν)
        C3 = cost_matrix(c, μ, ν)
        @test C1 ≈ pairwise(SqEuclidean(), μ.support, ν.support)
        @test C2 ≈ pairwise(SqEuclidean(), μ.support, ν.support)
        @test C3 ≈ pairwise(SqEuclidean(), μ.support, ν.support)
    end

    @testset "Creating cost matrices from matrices" begin
        N = 10
        M = 8
        μ = FiniteDiscreteMeasure(rand(N,3), rand(N))
        ν = FiniteDiscreteMeasure(rand(M,3), rand(M))
        c(x,y) = sum((x-y).^2)
        C1 = cost_matrix(SqEuclidean(), μ, ν)
        C2 = cost_matrix(sqeuclidean, μ, ν)
        C3 = cost_matrix(c, μ, ν)
        @test C1 ≈ pairwise(SqEuclidean(), μ.support, ν.support, dims=1)
        @test C2 ≈ pairwise(SqEuclidean(), μ.support, ν.support, dims=1)
        @test C3 ≈ pairwise(SqEuclidean(), μ.support, ν.support, dims=1)
    end
    @testset "Creating cost matrices from μ to itself" begin
        N = 10
        μ = FiniteDiscreteMeasure(rand(N,2), rand(N))
        c(x,y) = sqrt(sum((x-y).^2))
        C1 = cost_matrix(Euclidean(), μ, symmetric=true)
        C2 = cost_matrix(euclidean, μ, symmetric=true)
        C3 = cost_matrix(c, μ)
        @test C1 ≈ pairwise(Euclidean(), μ.support, μ.support, dims=1)
        @test C2 ≈ pairwise(Euclidean(), μ.support, μ.support, dims=1)
        @test C3 ≈ pairwise(Euclidean(), μ.support, μ.support, dims=1)
    end
end