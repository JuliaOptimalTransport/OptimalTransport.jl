using OptimalTransport

using Distances
using PythonOT: PythonOT

using Random
using Test

const POT = PythonOT

Random.seed!(100)

@testset "entropic.jl" begin
    @testset "sinkhorn" begin
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
            γ = sinkhorn(μ, ν, C, eps; maxiter=5_000, rtol=1e-9)
            γ_pot = POT.sinkhorn(μ, ν, C, eps; numItermax=5_000, stopThr=1e-9)
            @test γ_pot ≈ γ rtol = 1e-6

            # compute optimal transport cost
            c = sinkhorn2(μ, ν, C, eps; maxiter=5_000, rtol=1e-9)

            # with regularization term
            c_w_regularization = sinkhorn2(μ, ν, C, eps; maxiter=5_000, regularization=true)
            @test c_w_regularization ≈ c + eps * sum(x -> iszero(x) ? x : x * log(x), γ)

            # compare with POT
            c_pot = POT.sinkhorn2(μ, ν, C, eps; numItermax=5_000, stopThr=1e-9)[1]
            @test c_pot ≈ c

            # ensure that provided map is used and correct
            c2 = sinkhorn2(similar(μ), similar(ν), C, rand(); plan=γ)
            @test c2 ≈ c
            c2_w_regularization = sinkhorn2(
                similar(μ), similar(ν), C, eps; plan=γ, regularization=true
            )
            @test c2_w_regularization ≈ c_w_regularization
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
            γ = sinkhorn(μ, ν, C, eps; maxiter=5_000, rtol=1e-6)
            @test eltype(γ) === Float32

            γ_pot = POT.sinkhorn(μ, ν, C, eps; numItermax=5_000, stopThr=1e-6)
            @test Float32.(γ_pot) ≈ γ rtol = 1e-3

            # compute optimal transport cost
            c = sinkhorn2(μ, ν, C, eps; maxiter=5_000, rtol=1e-6)
            @test c isa Float32

            # with regularization term
            c_w_regularization = sinkhorn2(
                μ, ν, C, eps; maxiter=5_000, rtol=1e-6, regularization=true
            )
            @test c_w_regularization ≈ c + eps * sum(x -> iszero(x) ? x : x * log(x), γ)

            # compare with POT
            c_pot = POT.sinkhorn2(μ, ν, C, eps; numItermax=5_000, stopThr=1e-6)[1]
            @test Float32(c_pot) ≈ c rtol = 1e-3
        end

        @testset "deprecations" begin
            # create two uniform histograms
            μ = fill(1 / M, M)
            ν = fill(1 / N, N)

            # create random cost matrix
            C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

            # check `sinkhorn2`
            eps = 0.01
            c = sinkhorn2(μ, ν, C, eps; atol=1e-6)
            @test (@test_deprecated sinkhorn2(μ, ν, C, eps; tol=1e-6)) == c
            c = sinkhorn2(μ, ν, C, eps; check_convergence=5)
            @test (@test_deprecated sinkhorn2(μ, ν, C, eps; check_marginal_step=5)) == c

            # check `sinkhorn_gibbs
            K = @. exp(-C / eps)
            γ = OptimalTransport.sinkhorn_gibbs(μ, ν, K; atol=1e-6)
            @test (@test_deprecated OptimalTransport.sinkhorn_gibbs(μ, ν, K; tol=1e-6)) == γ
            γ = OptimalTransport.sinkhorn_gibbs(μ, ν, K; check_convergence=5)
            @test (@test_deprecated OptimalTransport.sinkhorn_gibbs(
                μ, ν, K; check_marginal_step=5
            )) == γ
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
            eps = 0.001
            γ = sinkhorn_stabilized(μ, ν, C, eps; maxiter=5_000)
            γ_pot = POT.sinkhorn(μ, ν, C, eps; method="sinkhorn_stabilized", numItermax=5_000)
            @test γ ≈ γ_pot atol = 1e-6
        end

        @testset "epsilon scaling" begin
            # create two uniform histograms
            μ = fill(1 / M, M)
            ν = fill(1 / N, N)

            # create random cost matrix
            C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

            # compute optimal transport map (Julia implementation + POT)
            eps = 0.001
            γ = sinkhorn_stabilized_epsscaling(μ, ν, C, eps; k=5, maxiter=1_000)
            γ_pot = POT.sinkhorn(μ, ν, C, eps; method="sinkhorn_stabilized", numItermax=5_000)
            @test γ ≈ γ_pot atol = 1e-6
        end
    end

    @testset "sinkhorn_barycenter" begin
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
        @test μ_interp ≈ μ_interp_pot
    end
end
