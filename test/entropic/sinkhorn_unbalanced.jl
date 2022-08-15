using OptimalTransport
using Distances
using PythonOT: PythonOT

using LinearAlgebra
using Random
using Test

const POT = PythonOT

Random.seed!(100)

@testset "sinkhorn_unbalanced.jl" begin
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
        function proxdivF1!(s, p, ε)
            return (s .= s .^ (ε / (ε + 0.4)) .* p .^ (0.4 / (ε + 0.4)) ./ s)
        end
        function proxdivF2!(s, p, ε)
            return (s .= s .^ (ε / (ε + 0.5)) .* p .^ (0.5 / (ε + 0.5)) ./ s)
        end
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

    @testset "unbalanced Sinkhorn divergences" begin
        μ = fill(1 / M, M)
        μ_spt = rand(1, M)
        ν = fill(1 / N, N)
        ν_spt = rand(1, N)
        ε = 0.01
        λ = 1.0
        Cμν = pairwise(SqEuclidean(), μ_spt, ν_spt; dims=2)
        Cμμ = pairwise(SqEuclidean(), μ_spt, μ_spt; dims=2)
        Cνν = pairwise(SqEuclidean(), ν_spt, ν_spt; dims=2)

        # check the symmetric terms 
        @test sinkhorn_unbalanced(μ, Cμμ, λ, ε) ≈ sinkhorn_unbalanced(μ, μ, Cμμ, λ, λ, ε) rtol =
            1e-4
        @test sinkhorn_unbalanced(ν, Cνν, λ, ε) ≈ sinkhorn_unbalanced(ν, ν, Cνν, λ, λ, ε) rtol =
            1e-4

        # check against balanced case 
        proxdivF!(s, p, ε) = (s .= p ./ s)
        @test sinkhorn_divergence_unbalanced(μ, ν, Cμν, Cμμ, Cνν, proxdivF!, ε) ≈
            sinkhorn_divergence(μ, ν, Cμν, Cμμ, Cνν, ε) rtol = 1e-4
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
        function proxdivF1(s, p)
            return s .^ (0.01 / (0.01 + 0.4)) .* p .^ (0.4 / (0.01 + 0.4)) ./ s
        end
        function proxdivF2(s, p)
            return s .^ (0.01 / (0.01 + 0.5)) .* p .^ (0.5 / (0.01 + 0.5)) ./ s
        end
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
