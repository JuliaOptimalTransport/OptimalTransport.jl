using OptimalTransport

using Distances
using Distributions

using Random
using Test

Random.seed!(100)

@testset "wasserstein.jl" begin
    @testset "p2distance" begin
        for metric in (Euclidean(), Euclidean(0.01), TotalVariation())
            @test OptimalTransport.p2distance(metric, Val(1)) === metric
        end

        @test OptimalTransport.p2distance(Euclidean(), Val(2)) == SqEuclidean()
        @test OptimalTransport.p2distance(Euclidean(0.01), Val(2)) == SqEuclidean(0.01)

        p = randexp()
        x = randn(10)
        y = randn(10)
        for metric in (Euclidean(), TotalVariation())
            for _p in (p, Val(p))
                pmetric = OptimalTransport.p2distance(metric, _p)
                @test pmetric(x, y) ≈ metric(x, y)^p
            end
        end
    end

    @testset "prt" begin
        x = randexp()
        for p in (1, 2, 3, randexp())
            @test OptimalTransport.prt(x, p) ≈ x^(1 / p)
            @test OptimalTransport.prt(x, Val(p)) ≈ x^(1 / p)
        end
    end

    @testset "wasserstein" begin
        μ = Normal(randn(), randexp())
        ν = Normal(randn(), randexp())
        for p in (1, 2, 3, randexp()), metric in (Euclidean(), TotalVariation())
            for _p in (p, Val(p))
                # without additional keyword arguments
                w = wasserstein(μ, ν; p=_p, metric=metric)
                @test w ≈ ot_cost((x, y) -> metric(x, y)^p, μ, ν)^(1 / p)

                # with pre-computed plan (random `ν` ensures that plan is used)
                T = ot_plan((x, y) -> metric(x, y)^p, μ, ν)
                w2 = wasserstein(μ, Normal(randn(), rand()); p=_p, metric=metric, plan=T)
                @test w ≈ w2
            end
        end

        # check that `Euclidean` is the default `metric`
        for p in (1, 2, 3, randexp()), _p in (p, Val(p))
            w = wasserstein(μ, ν; p=_p)
            @test w ≈ wasserstein(μ, ν; p=_p, metric=Euclidean())
        end

        # check that `Val(1)` is the default `p`
        for metric in (Euclidean(), TotalVariation())
            w = wasserstein(μ, ν; metric=metric)
            @test w ≈ wasserstein(μ, ν; p=Val(1), metric=metric)
        end
    end

    @testset "squared2wasserstein" begin
        μ = Normal(randn(), randexp())
        ν = Normal(randn(), randexp())
        for metric in (Euclidean(), TotalVariation())
            # without additional keyword arguments
            w = squared2wasserstein(μ, ν; metric=metric)
            @test w ≈ ot_cost((x, y) -> metric(x, y)^2, μ, ν)

            # with pre-computed plan (random `ν` ensures that plan is used)
            T = ot_plan((x, y) -> metric(x, y)^2, μ, ν)
            w2 = squared2wasserstein(μ, Normal(randn(), rand()); metric=metric, plan=T)
            @test w ≈ w2
        end

        # check that `Euclidean` is the default `metric`
        w = squared2wasserstein(μ, ν)
        @test w ≈ squared2wasserstein(μ, ν; metric=Euclidean())
    end
end
