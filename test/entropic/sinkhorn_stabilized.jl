using OptimalTransport

using Distances
using ForwardDiff
using LogExpFunctions
using PythonOT: PythonOT

using LinearAlgebra
using Random
using Test

const POT = PythonOT

Random.seed!(100)

@testset "sinkhorn_stabilized.jl" begin
    # size of source and target
    M = 250
    N = 200

    # create two random histograms
    μ = normalize!(rand(M), 1)
    ν = normalize!(rand(N), 1)

    # create random cost matrix
    C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

    # regularization parameter
    ε = 0.1

    @testset "example" begin
        # compute optimal transport plan and optimal transport cost
        γ = sinkhorn(μ, ν, C, ε, SinkhornStabilized())
        c = sinkhorn2(μ, ν, C, ε, SinkhornStabilized())

        # check that plan and cost are consistent
        @test c ≈ dot(γ, C)

        # compare with POT
        γ_pot = POT.sinkhorn(μ, ν, C, ε; method="sinkhorn_stabilized", stopThr=1e-16)
        c_pot = POT.sinkhorn2(μ, ν, C, ε; method="sinkhorn_stabilized", stopThr=1e-16)[1]
        @test γ_pot ≈ γ
        @test c_pot ≈ c

        # compute optimal transport cost with regularization term
        c_w_regularization = sinkhorn2(
            μ, ν, C, ε, SinkhornStabilized(); regularization=true
        )
        @test c_w_regularization ≈ c + ε * sum(x -> iszero(x) ? x : x * log(x), γ)

        # ensure that provided plan is used and correct
        c2 = sinkhorn2(similar(μ), similar(ν), C, rand(), SinkhornStabilized(); plan=γ)
        @test c2 ≈ c
        c2_w_regularization = sinkhorn2(
            similar(μ), similar(ν), C, ε, SinkhornStabilized(); plan=γ, regularization=true
        )
        @test c2_w_regularization ≈ c_w_regularization

        # batches of histograms
        d = 10
        for (size2_μ, size2_ν) in
            (((), (d,)), ((1,), (d,)), ((d,), ()), ((d,), (1,)), ((d,), (d,)))
            # generate batches of histograms
            μ_batch = repeat(μ, 1, size2_μ...)
            ν_batch = repeat(ν, 1, size2_ν...)

            # compute optimal transport plan and check that it is consistent with the
            # plan for individual histograms
            γ_all = sinkhorn(μ_batch, ν_batch, C, ε, SinkhornStabilized())
            @test size(γ_all) == (M, N, d)
            @test all(view(γ_all,:,:,i) ≈ γ for i in axes(γ_all, 3))

            # compute optimal transport cost and check that it is consistent with the
            # cost for individual histograms
            c_all = sinkhorn2(μ_batch, ν_batch, C, ε, SinkhornStabilized())
            @test size(c_all) == (d,)
            @test all(x ≈ c for x in c_all)
        end
    end

    @testset "infinite absorption tolerance" begin
        # compute optimal transport plan with infinite absorption tolerance
        alg = SinkhornStabilized(; absorb_tol=Inf)
        γ = sinkhorn(μ, ν, C, ε, alg; maxiter=100)
        c = sinkhorn(μ, ν, C, ε, alg; maxiter=100)

        # compare with regular Sinkhorn algorithm
        γ_sinkhorn = sinkhorn(μ, ν, C, ε; maxiter=100)
        c_sinkhorn = sinkhorn(μ, ν, C, ε; maxiter=100)
        @test γ_sinkhorn ≈ γ
        @test c_sinkhorn ≈ c
    end

    @testset "AD" begin
        # compute gradients with respect to source and target marginals separately and
        # together
        ForwardDiff.gradient(zeros(N)) do xs
            return sinkhorn2(μ, softmax(xs), C, ε, SinkhornStabilized())
        end
        ForwardDiff.gradient(zeros(M)) do xs
            return sinkhorn2(softmax(xs), ν, C, ε, SinkhornStabilized())
        end
        ForwardDiff.gradient(zeros(M + N)) do xs
            return sinkhorn2(
                softmax(xs[1:M]), softmax(xs[(M + 1):end]), C, ε, SinkhornStabilized()
            )
        end
    end

    @testset "deprecations" begin
        γ = sinkhorn(μ, ν, C, ε, SinkhornStabilized(; absorb_tol=10))
        @test (@test_deprecated sinkhorn_stabilized(μ, ν, C, ε; absorb_tol=10)) == γ

        γ = sinkhorn(μ, ν, C, ε, SinkhornStabilized(); atol=1e-6)
        @test (@test_deprecated sinkhorn_stabilized(μ, ν, C, ε; tol=1e-6)) == γ

        γ = sinkhorn(μ, ν, C, ε, SinkhornStabilized())
        u, v = @test_deprecated sinkhorn_stabilized(μ, ν, C, ε; return_duals=true)
        @test exp.(-(C .- u .- v') / ε) ≈ γ
    end
end
