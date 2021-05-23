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
    # batch kernel reduction 
    @testset "batch" begin
        # create two sets of batch histograms 
        d = 10
        μ = rand(Float64, (M, d))
        μ = μ./sum(μ, dims = 1)
        ν = rand(Float64, (N, d))
        ν = ν./sum(ν, dims = 1)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.01
        γ_all = sinkhorn(μ, ν, C, eps)
        γ_pot = [POT.sinkhorn(μ[:, i], ν[:, i], C, eps) for i = 1:d]
        @test maximum([norm(γ_all[i] - γ_pot[i], Inf) for i = 1:d]) < 1e-9

        c_all = sinkhorn2(μ, ν, C, eps)
        c_pot = [POT.sinkhorn2(μ[:, i], ν[:, i], C, eps) for i = 1:d]
        @test c_all ≈ c_pot atol = 1e-9 norm = (x -> norm(x, Inf))
    end

    # batch kernel reduction 
    @testset "batch" begin
        # create two sets of batch histograms 
        d = 10
        μ = rand(Float64, (M, d))
        μ = μ./sum(μ, dims = 1)
        ν = rand(Float64, (N, d))
        ν = ν./sum(ν, dims = 1)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.01
        γ_all = sinkhorn(μ, ν, C, eps)
        γ_pot = [POT.sinkhorn(μ[:, i], ν[:, i], C, eps) for i = 1:d]
        @test maximum([norm(γ_all[i] - γ_pot[i], Inf) for i = 1:d]) < 1e-9

        c_all = sinkhorn2(μ, ν, C, eps)
        c_pot = [POT.sinkhorn2(μ[:, i], ν[:, i], C, eps) for i = 1:d]
        @test c_all ≈ c_pot atol = 1e-9 norm = (x -> norm(x, Inf))
    end

    # computation on the GPU
    if CUDA.functional()
        @testset "CUDA" begin
            # create two uniform histograms
            μ = CUDA.fill(Float32(1 / M), M)
            ν = CUDA.fill(Float32(1 / N), N)

            # create random cost matrix
            C = abs2.(CUDA.rand(M) .- CUDA.rand(1, N))

            # compute optimal transport map
            eps = 0.01f0
            γ = sinkhorn(μ, ν, C, eps)
            @test γ isa CuArray{Float32}

            # compute optimal transport cost
            c = sinkhorn2(μ, ν, C, eps)
            @test c isa Float32
        end
    end
end

@testset "unbalanced transport" begin
    M = 250
    N = 200

    @testset "example" begin
        μ = fill(1 / N, M)
        ν = fill(1 / N, N)

        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute optimal transport plan
        eps = 0.01
        lambda = 1
        γ = sinkhorn_unbalanced(μ, ν, C, lambda, lambda, eps)

        # compare with POT
        γ_pot = POT.sinkhorn_unbalanced(μ, ν, C, eps, lambda; stopThr=1e-9)
        @test γ_pot ≈ γ

        # compute optimal transport cost
        c = sinkhorn_unbalanced2(μ, ν, C, lambda, lambda, eps; maxiter=5_000)

        # compare with POT
        c_pot = POT.sinkhorn_unbalanced2(
            μ, ν, C, eps, lambda; numItermax=5_000, stopThr=1e-9
        )[1]
        @test c_pot ≈ c

        # ensure that provided plan is used
        c2 = sinkhorn_unbalanced2(similar(μ), similar(ν), C, rand(), rand(), rand(); plan=γ)
        @test c2 ≈ c
    end

    @testset "proxdiv operators" begin
        μ = fill(1 / N, M)
        ν = fill(1 / N, N)
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute optimal transport plan and cost with real-valued parameters
        eps = 0.01
        lambda1 = 0.4
        lambda2 = 0.5
        γ = sinkhorn_unbalanced(μ, ν, C, lambda1, lambda2, eps)
        c = sinkhorn_unbalanced2(μ, ν, C, lambda1, lambda2, eps)
        @test c ≈ dot(γ, C)

        # compute optimal transport plan and cost with manual "proxdiv" functions
        proxdivF1!(s, p, ε) = (s .= s .^ (ε / (ε + 0.4)) .* p .^ (0.4 / (ε + 0.4)) ./ s)
        proxdivF2!(s, p, ε) = (s .= s .^ (ε / (ε + 0.5)) .* p .^ (0.5 / (ε + 0.5)) ./ s)
        γ_proxdiv = sinkhorn_unbalanced(μ, ν, C, proxdivF1!, proxdivF2!, eps)
        c_proxdiv = sinkhorn_unbalanced2(μ, ν, C, proxdivF1!, proxdivF2!, eps)
        @test γ_proxdiv ≈ γ
        @test c_proxdiv ≈ c
    end

    @testset "consistency with balanced OT" begin
        μ = fill(1 / M, M)
        ν = fill(1 / N, N)
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute optimal transport plan and cost with manual "proxdiv" functions for
        # balanced OT
        ε = 0.01
        proxdivF!(s, p, ε) = (s .= p ./ s)
        γ = sinkhorn_unbalanced(μ, ν, C, proxdivF!, proxdivF!, ε)
        c = sinkhorn_unbalanced2(μ, ν, C, proxdivF!, proxdivF!, ε)
        @test c ≈ dot(γ, C)

        # compute optimal transport plan and cost for balanced OT
        γ_balanced = sinkhorn(μ, ν, C, ε)
        c_balanced = sinkhorn2(μ, ν, C, ε)
        @test γ_balanced ≈ γ rtol = 1e-4
        @test c_balanced ≈ c rtol = 1e-4
    end

    @testset "deprecations" begin
        μ = fill(1 / N, M)
        ν = fill(1 / N, N)
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute optimal transport plan and cost with real-valued parameters
        ε = 0.01
        λ1 = 0.4
        λ2 = 0.5
        γ = sinkhorn_unbalanced(μ, ν, C, λ1, λ2, ε)
        c = sinkhorn_unbalanced2(μ, ν, C, λ1, λ2, ε)

        # compute optimal transport plan and cost with manual "proxdiv" functions
        # as keyword arguments
        proxdivF1(s, p) = s .^ (0.01 / (0.01 + 0.4)) .* p .^ (0.4 / (0.01 + 0.4)) ./ s
        proxdivF2(s, p) = s .^ (0.01 / (0.01 + 0.5)) .* p .^ (0.5 / (0.01 + 0.5)) ./ s
        γ_proxdiv = @test_deprecated sinkhorn_unbalanced(
            μ, ν, C, rand(), rand(), ε; proxdiv_F1=proxdivF1, proxdiv_F2=proxdivF2
        )
        c_proxdiv = @test_deprecated sinkhorn_unbalanced2(
            μ, ν, C, rand(), rand(), ε; proxdiv_F1=proxdivF1, proxdiv_F2=proxdivF2
        )
        @test γ_proxdiv ≈ γ
        @test c_proxdiv ≈ c

        # deprecated `tol` keyword argument
        γ = sinkhorn_unbalanced(μ, ν, C, λ1, λ2, ε; atol=1e-7)
        c = sinkhorn_unbalanced2(μ, ν, C, λ1, λ2, ε; atol=1e-7)
        γ_tol = @test_deprecated sinkhorn_unbalanced(μ, ν, C, λ1, λ2, ε; tol=1e-7)
        c_tol = @test_deprecated sinkhorn_unbalanced2(μ, ν, C, λ1, λ2, ε; tol=1e-7)
        @test γ_tol == γ
        @test c_tol == c

        # deprecated `max_iter` keyword argument
        γ = sinkhorn_unbalanced(μ, ν, C, λ1, λ2, ε; maxiter=50)
        c = sinkhorn_unbalanced2(μ, ν, C, λ1, λ2, ε; maxiter=50)
        γ_max_iter = @test_deprecated sinkhorn_unbalanced(μ, ν, C, λ1, λ2, ε; max_iter=50)
        c_max_iter = @test_deprecated sinkhorn_unbalanced2(μ, ν, C, λ1, λ2, ε; max_iter=50)
        @test γ_max_iter == γ
        @test c_max_iter == c
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
        eps = 0.0025
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
        μ_all = hcat(μ1, μ2)
        # create cost matrix
        C = pairwise(SqEuclidean(), support'; dims=2)

        # compute Sinkhorn barycenter (Julia implementation + POT)
        eps = 0.01
        μ_interp = sinkhorn_barycenter(μ_all, [C, C], eps, [0.5, 0.5])
        μ_interp_pot = POT.barycenter(μ_all', C, eps; weights=[0.5, 0.5], stopThr=1e-9)
        @test norm(μ_interp - μ_interp_pot, Inf) < 1e-9
        @test norm(μ_interp - μ_interp_batch, Inf) < 1e-9
    end
end
