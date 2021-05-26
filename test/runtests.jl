using OptimalTransport

using Distances
using PythonOT: PythonOT
using Tulip
using MathOptInterface
using Distributions
using SparseArrays

using LinearAlgebra
using Random
using Test

const MOI = MathOptInterface
const POT = PythonOT

Random.seed!(100)

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
    # Continuous Case
    μ = Normal(0, 1)
    ν = Normal(5, 2)
    γ = otplan(sqeuclidean, μ, ν)

    @test otcost(sqeuclidean, μ, ν) ≈ 26 atol = 1e-5
    @test otcost(sqeuclidean, μ, ν; plan=γ) ≈ 26 atol = 1e-5

    # Discrete Case
    # n, m = 10, 15

    # μ_support = rand(n)
    # ν_support = rand(m)
    # μ_probs = rand(n)
    # ν_probs = rand(m)
    # μ_probs ./= sum(μ_probs)
    # ν_probs ./= sum(ν_probs)

    # μ = DiscreteNonParametric(μ_support, μ_probs)
    # ν = DiscreteNonParametric(ν_support, ν_probs)

    # # new version of StatsBase also has functoin pairwise,
    # # which conflicts with Distances.pairwise
    # C = pairwise(sqeuclidean, μ.support, ν.support)

    # lp = Tulip.Optimizer()
    # cost_simplex = emd2(μ.p, ν.p, C, lp)
    # cost_1d = otcost(sqeuclidean, μ, ν)

    # P = emd(μ.p, ν.p, C, lp)
    # γ = otplan(sqeuclidean, μ, ν)

    # @test cost_1d ≈ cost_simplex atol = 1e-6
    # @test γ ≈ P atol = 1e-5
    # @test otcost(sqeuclidean, μ, ν; plan=γ) ≈ cost_simplex atol = 1e-6
end

@testset "entropically regularized transport" begin
    M = 250
    N = 200

    @testset "example" begin
        # create two uniform histograms
        μ = fill(1 / M, M)
        ν = fill(1 / N, N)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.01
        γ = sinkhorn(μ, ν, C, eps; maxiter=5_000)
        γ_pot = POT.sinkhorn(μ, ν, C, eps; numItermax=5_000, stopThr=1e-9)
        @test norm(γ - γ_pot, Inf) < 1e-9

        # compute optimal transport cost (Julia implementation + POT)
        c = sinkhorn2(μ, ν, C, eps; maxiter=5_000)
        c_pot = POT.sinkhorn2(μ, ν, C, eps; numItermax=5_000, stopThr=1e-9)[1]
        @test c ≈ c_pot atol = 1e-9

        # ensure that provided map is used
        c2 = sinkhorn2(similar(μ), similar(ν), C, rand(); plan=γ)
        @test c2 ≈ c
    end

    # different element type
    @testset "Float32" begin
        # create two uniform histograms
        μ = fill(Float32(1 / M), M)
        ν = fill(Float32(1 / N), N)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(Float32, 1, M), rand(Float32, 1, N); dims=2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.01f0
        γ = sinkhorn(μ, ν, C, eps; maxiter=5_000)
        @test eltype(γ) === Float32

        γ_pot = POT.sinkhorn(μ, ν, C, eps; numItermax=5_000, stopThr=1e-9)
        @test norm(γ - γ_pot, Inf) < Base.eps(Float32)

        # compute optimal transport cost (Julia implementation + POT)
        c = sinkhorn2(μ, ν, C, eps; maxiter=5_000)
        @test c isa Float32

        c_pot = POT.sinkhorn2(μ, ν, C, eps; numItermax=5_000, stopThr=1e-9)[1]
        @test c ≈ c_pot atol = Base.eps(Float32)
    end
end

@testset "unbalanced transport" begin
    M = 250
    N = 200
    @testset "example" begin
        μ = fill(1 / N, M)
        ν = fill(1 / N, N)

        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        eps = 0.01
        lambda = 1
        γ = sinkhorn_unbalanced(μ, ν, C, lambda, lambda, eps)
        γ_pot = POT.sinkhorn_unbalanced(μ, ν, C, eps, lambda; stopThr=1e-9)

        # compute optimal transport map
        @test norm(γ - γ_pot, Inf) < 1e-9

        c = sinkhorn_unbalanced2(μ, ν, C, lambda, lambda, eps; max_iter=5_000)
        c_pot = POT.sinkhorn_unbalanced2(
            μ, ν, C, eps, lambda; numItermax=5_000, stopThr=1e-9
        )[1]

        @test c ≈ c_pot atol = 1e-9

        # ensure that provided map is used
        c2 = sinkhorn_unbalanced2(similar(μ), similar(ν), C, rand(), rand(), rand(); plan=γ)
        @test c2 ≈ c
    end
end

@testset "stabilized sinkhorn" begin
    M = 250
    N = 200

    @testset "example" begin
        # create two uniform histograms
        μ = fill(1 / M, M)
        ν = fill(1 / N, N)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.01
        γ = sinkhorn_stabilized(μ, ν, C, eps)
        γ_pot = POT.sinkhorn(μ, ν, C, eps; method="sinkhorn_stabilized")
        @test norm(γ - γ_pot, Inf) < 1e-9
    end
end

@testset "quadratic optimal transport" begin
    M = 250
    N = 200
    @testset "example" begin
        # create two uniform histograms
        μ = fill(1 / M, M)
        ν = fill(1 / N, N)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.25
        γ = quadreg(μ, ν, C, eps)
        γ_pot = POT.Smooth.smooth_ot_dual(μ, ν, C, eps; stopThr=1e-9)
        # need to use a larger tolerance here because of a quirk with the POT solver 
        @test norm(γ - γ_pot, Inf) < 1e-4
    end
end

@testset "sinkhorn barycenter" begin
    @testset "example" begin
        # set up support
        support = range(-1; stop=1, length=250)
        μ1 = exp.(-(support .+ 0.5) .^ 2 ./ 0.1^2)
        μ1 ./= sum(μ1)
        μ2 = exp.(-(support .- 0.5) .^ 2 ./ 0.1^2)
        μ2 ./= sum(μ2)
        μ_all = hcat(μ1, μ2)'

        # create cost matrix
        C = pairwise(SqEuclidean(), support'; dims=2)

        # compute Sinkhorn barycenter (Julia implementation + POT)
        eps = 0.01
        μ_interp = sinkhorn_barycenter(μ_all, [C, C], eps, [0.5, 0.5])
        μ_interp_pot = POT.barycenter(μ_all', C, eps; weights=[0.5, 0.5], stopThr=1e-9)
        @test norm(μ_interp - μ_interp_pot, Inf) < 1e-9
    end
end
