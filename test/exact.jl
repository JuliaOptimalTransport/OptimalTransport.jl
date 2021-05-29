using OptimalTransport

using Distances
using PythonOT: PythonOT
using Tulip
using MathOptInterface
using Distributions

using LinearAlgebra
using Random

const MOI = MathOptInterface
const POT = PythonOT

Random.seed!(100)

@testset "exact.jl" begin
    @testset "Earth-Movers Distance" begin
        M = 200
        N = 250
        μ = rand(M)
        ν = rand(N)
        μ ./= sum(μ)
        ν ./= sum(ν)

        @testset "example" begin
            # create random cost matrix
            C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

            # compute optimal transport map and cost with POT
            pot_P = POT.emd(μ, ν, C)
            pot_cost = POT.emd2(μ, ν, C)

            # compute optimal transport map and cost with Tulip
            lp = Tulip.Optimizer()
            P = emd(μ, ν, C, lp)
            @test size(C) == size(P)
            @test MOI.get(lp, MOI.TerminationStatus()) == MOI.OPTIMAL
            @test maximum(abs, P .- pot_P) < 1e-2

            lp = Tulip.Optimizer()
            cost = emd2(μ, ν, C, lp)
            @test dot(C, P) ≈ cost atol = 1e-5
            @test MOI.get(lp, MOI.TerminationStatus()) == MOI.OPTIMAL
            @test cost ≈ pot_cost atol = 1e-5
        end

        @testset "pre-computed plan" begin
            # create random cost matrix
            C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

            # compute optimal transport map
            P = emd(μ, ν, C, Tulip.Optimizer())

            # do not use μ and ν to ensure that provided map is used
            cost = emd2(similar(μ), similar(ν), C, Tulip.Optimizer(); plan=P)
            @test cost ≈ emd2(μ, ν, C, Tulip.Optimizer())
        end

        # https://github.com/JuliaOptimalTransport/OptimalTransport.jl/issues/71
        @testset "cost matrix with integers" begin
            C = pairwise(SqEuclidean(), rand(1:10, 1, M), rand(1:10, 1, N); dims=2)
            emd2(μ, ν, C, Tulip.Optimizer())
        end
    end

    @testset "1D Optimal Transport for Convex Cost" begin
        @testset "continuous distributions" begin
            # two normal distributions (has analytical solution)
            μ = Normal(randn(), rand())
            ν = Normal(randn(), rand())

            # compute OT plan
            γ = ot_plan(sqeuclidean, μ, ν)
            x = randn()
            @test γ(x) ≈ quantile(ν, cdf(μ, x))

            # compute OT cost
            c = ot_cost(sqeuclidean, μ, ν)
            @test c ≈ (mean(μ) - mean(ν))^2 + (std(μ) - std(ν))^2

            # do not use ν to ensure that the provided plan is used
            @test ot_cost(sqeuclidean, μ, Normal(randn(), rand()); plan=γ) ≈ c
        end

        @testset "semidiscrete case" begin
            μ = Normal(randn(), rand())
            νprobs = rand(30)
            νprobs ./= sum(νprobs)
            ν = Categorical(νprobs)

            # compute OT plan
            γ = ot_plan(euclidean, μ, ν)
            x = randn()
            @test γ(x) ≈ quantile(ν, cdf(μ, x))

            # compute OT cost, without and with provided plan
            # do not use ν in the second case to ensure that the provided plan is used
            c = ot_cost(euclidean, μ, ν)
            @test ot_cost(euclidean, μ, Categorical(reverse(νprobs)); plan=γ) ≈ c

            # check that OT cost is consistent with OT cost of a discretization
            m = 500
            xs = rand(μ, m)
            μdiscrete = fill(1 / m, m)
            C = pairwise(Euclidean(), xs', (1:length(νprobs))'; dims=2)
            c2 = emd2(μdiscrete, νprobs, C, Tulip.Optimizer())
            @test c2 ≈ c rtol = 1e-1
        end
    end
end
