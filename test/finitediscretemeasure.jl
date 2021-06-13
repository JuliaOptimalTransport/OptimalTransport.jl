using Distributions: DiscreteNonParametric
using OptimalTransport
using Distributions
using Random

Random.seed!(100)

@testset "finitediscretemeasure.jl" begin
    @testset "Univariate Finite Discrete Measure" begin
        n = 100
        μsupp = rand(n)
        νsupp = rand(n, 1)
        μ = FiniteDiscreteMeasure(μsupp)
        ν = FiniteDiscreteMeasure(νsupp, rand(n))
        # check if it assigns equal probabilities to all entries
        @test μ.p ≈ ones(n) ./ n
        @test probs(μ) ≈ ones(n) ./ n
        # check if it probabilities sum to 1
        @test sum(ν.p) ≈ 1
        @test sum(probs(ν)) ≈ 1
        # check if probabilities are all positive (non-negative)
        @test all(ν.p .>= 0)
        @test all(probs(ν) .>= 0)
        # check if it assigns to DiscreteNonParametric when Vector/Matrix is 1D
        @test typeof(μ) <: DiscreteNonParametric
        @test typeof(ν) <: DiscreteNonParametric
        # check if support is correctly assinged
        @test sort(μsupp) == μ.support
        @test sort(μsupp) == support(μ)
        @test sort(vec(νsupp)) == ν.support
        @test sort(vec(νsupp)) == support(ν)
    end
    @testset "Multivariate Finite Discrete Measure" begin
        n = 10
        m = 3
        μsupp = rand(n, m)
        νsupp = rand(n, m)
        μ = FiniteDiscreteMeasure(μsupp)
        ν = FiniteDiscreteMeasure(νsupp, rand(n))
        # check if it assigns equal probabilities to all entries
        @test μ.p ≈ ones(n) ./ n
        @test probs(μ) ≈ ones(n) ./ n
        # check if it probabilities sum to 1
        @test sum(ν.p) ≈ 1
        @test sum(probs(ν)) ≈ 1
        # check if probabilities are all positive (non-negative)
        @test all(ν.p .>= 0)
        @test all(probs(ν) .>= 0)
        # check if support is correctly assinged
        @test μsupp == μ.support
        @test μsupp == support(μ)
        @test νsupp == ν.support
        @test νsupp == support(ν)
    end
end
