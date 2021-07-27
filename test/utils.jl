using OptimalTransport

using LinearAlgebra
using Random
using Test
using Distributions

Random.seed!(100)

@testset "utils.jl" begin
    @testset "add_singleton" begin
        x = rand(3)
        y = @inferred(OptimalTransport.add_singleton(x, Val(1)))
        @test size(y) == (1, length(x))
        @test vec(y) == x

        y = @inferred(OptimalTransport.add_singleton(x, Val(2)))
        @test size(y) == (length(x), 1)
        @test vec(y) == x

        x = rand(3, 4)
        y = @inferred(OptimalTransport.add_singleton(x, Val(1)))
        @test size(y) == (1, size(x, 1), size(x, 2))
        @test vec(y) == vec(x)

        y = @inferred(OptimalTransport.add_singleton(x, Val(2)))
        @test size(y) == (size(x, 1), 1, size(x, 2))
        @test vec(y) == vec(x)

        y = @inferred(OptimalTransport.add_singleton(x, Val(3)))
        @test size(y) == (size(x, 1), size(x, 2), 1)
        @test vec(y) == vec(x)
    end

    @testset "dot_matwise" begin
        l, m, n = 4, 5, 3
        x = rand(l, m)
        y = rand(l, m)
        @test OptimalTransport.dot_matwise(x, y) == dot(x, y)

        y = rand(l, m, n)
        @test OptimalTransport.dot_matwise(x, y) ≈
              mapreduce(vcat, (view(y, :, :, i) for i in axes(y, 3))) do yi
            dot(x, yi)
        end
        @test OptimalTransport.dot_matwise(y, x) == OptimalTransport.dot_matwise(x, y)
    end

    @testset "checksize" begin
        for xsize in ((5,), (5, 1), (5, 4)), ysize in ((3,), (3, 1), (3, 6))
            x = rand(xsize...)
            y = rand(ysize...)
            z = rand(xsize[1], ysize[1])
            OptimalTransport.checksize(x, y, z)
        end
    end

    @testset "checksize2" begin
        x = rand(5)
        y = rand(10)
        @test OptimalTransport.checksize2(x, y) === ()

        d = 4
        for (size2_x, size2_y) in
            (((), (d,)), ((1,), (d,)), ((d,), ()), ((d,), (1,)), ((d,), (d,)))
            x = rand(5, size2_x...)
            y = rand(10, size2_y...)
            @test OptimalTransport.checksize2(x, y) == (d,)
        end

        x = rand(5, 4)
        y = rand(10, 3)
        @test_throws DimensionMismatch OptimalTransport.checksize2(x, y)
    end

    @testset "checkbalanced" begin
        mass = rand()

        x1 = normalize!(rand(20), 1)
        x1 .*= mass
        y1 = normalize!(rand(30), 1)
        y1 .*= mass
        @test OptimalTransport.checkbalanced(x1, y1) === nothing
        @test OptimalTransport.checkbalanced(y1, x1) === nothing
        @test_throws ArgumentError OptimalTransport.checkbalanced(rand() .* x1, y1)
        @test_throws ArgumentError OptimalTransport.checkbalanced(x1, rand() .* y1)

        y2 = rand(30, 5)
        y2 .*= mass ./ sum(y2; dims=1)
        @test OptimalTransport.checkbalanced(x1, y2) === nothing
        @test OptimalTransport.checkbalanced(y2, x1) === nothing
        @test_throws ArgumentError OptimalTransport.checkbalanced(rand() .* x1, y2)
        @test_throws ArgumentError OptimalTransport.checkbalanced(
            x1, y2 .* hcat(rand(), ones(1, size(y2, 2) - 1))
        )

        x2 = rand(20, 5)
        x2 .*= mass ./ sum(x2; dims=1)
        @test OptimalTransport.checkbalanced(x2, y2) === nothing
        @test OptimalTransport.checkbalanced(y2, x2) === nothing
        @test_throws ArgumentError OptimalTransport.checkbalanced(
            x2 .* hcat(ones(1, size(x2, 2) - 1), rand()), y2
        )
        @test_throws ArgumentError OptimalTransport.checkbalanced(
            x2, y2 .* hcat(rand(), ones(1, size(y2, 2) - 1))
        )
    end

    @testset "A_batched_mul_B!" begin
        l, m, n = 5, 10, 4
        A = rand(l, m)
        b = rand(m)
        c = rand(l)
        OptimalTransport.A_batched_mul_B!(c, A, b)
        @test c == A * b

        B = rand(m, n)
        C = rand(l, n)
        OptimalTransport.A_batched_mul_B!(C, A, B)
        @test C == A * B

        As = rand(l, m, n)
        OptimalTransport.A_batched_mul_B!(C, As, B)
        @test all(C[:, i] ≈ As[:, :, i] * B[:, i] for i in 1:n)
    end

    @testset "At_batched_mul_B!" begin
        l, m, n = 5, 10, 4
        A = rand(l, m)
        b = rand(l)
        c = rand(m)
        OptimalTransport.At_batched_mul_B!(c, A, b)
        @test c == transpose(A) * b

        B = rand(l, n)
        C = rand(m, n)
        OptimalTransport.At_batched_mul_B!(C, A, B)
        @test C == transpose(A) * B

        As = rand(l, m, n)
        OptimalTransport.At_batched_mul_B!(C, As, B)
        @test all(C[:, i] ≈ transpose(As[:, :, i]) * B[:, i] for i in 1:n)
    end

    @testset "FiniteDiscreteMeasure" begin
        @testset "Univariate Finite Discrete Measure" begin
            n = 100
            m = 80
            μsupp = rand(n)
            νsupp = rand(m)
            μprobs = normalize!(rand(n), 1)

            μ = OptimalTransport.discretemeasure(μsupp, μprobs)
            ν = OptimalTransport.discretemeasure(νsupp)
            # check if it vectors are indeed probabilities
            @test isprobvec(μ.p)
            @test isprobvec(probs(μ))
            @test ν.p == ones(m) ./ m
            @test probs(ν) == ones(m) ./ m

            # check if it assigns to DiscreteNonParametric when Vector/Matrix is 1D
            @test μ isa DiscreteNonParametric
            @test ν isa DiscreteNonParametric

            # check if support is correctly assinged
            @test sort(μsupp) == μ.support
            @test sort(μsupp) == support(μ)
            @test sort(vec(νsupp)) == ν.support
            @test sort(vec(νsupp)) == support(ν)
        end
        @testset "Multivariate Finite Discrete Measure" begin
            n = 10
            m = 3
            μsupp = [rand(m) for i in 1:n]
            νsupp = [rand(m) for i in 1:n]
            μprobs = normalize!(rand(n), 1)
            μ = OptimalTransport.discretemeasure(μsupp, μprobs)
            ν = OptimalTransport.discretemeasure(νsupp)
            # check if it vectors are indeed probabilities
            @test isprobvec(μ.p)
            @test isprobvec(probs(μ))
            @test ν.p == ones(n) ./ n
            @test probs(ν) == ones(n) ./ n

            # check if support is correctly assinged
            @test μsupp == μ.support
            @test μsupp == support(μ)
            @test νsupp == ν.support
            @test νsupp == support(ν)
        end
    end
end
