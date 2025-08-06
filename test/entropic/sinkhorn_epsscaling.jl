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

@testset "sinkhorn_epsscaling.jl" begin
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
        # compute optimal transport plan and cost with POT
        γ_pot = POT.sinkhorn(μ, ν, C, ε; method="sinkhorn_stabilized", stopThr=1e-16)
        c_pot = POT.sinkhorn2(μ, ν, C, ε; method="sinkhorn_stabilized", stopThr=1e-16)[1]

        for alg in (SinkhornGibbs(), SinkhornStabilized())
            # compute optimal transport plan and cost
            γ = sinkhorn(μ, ν, C, ε, SinkhornEpsilonScaling(alg))
            c = sinkhorn2(μ, ν, C, ε, SinkhornEpsilonScaling(alg))

            # check that plan and cost are consistent
            @test c ≈ dot(γ, C)

            # compare with Sinkhorn algorithm without ε-scaling
            γ_wo_epsscaling = sinkhorn(μ, ν, C, ε, alg)
            c_wo_epsscaling = sinkhorn2(μ, ν, C, ε, alg)
            @test γ ≈ γ_wo_epsscaling
            @test c ≈ c_wo_epsscaling

            # compare with POT
            @test γ ≈ γ_pot
            @test c ≈ c_pot
        end
    end

    @testset "AD" begin
        # compute gradients with respect to source and target marginals separately and
        # together
        for alg in (SinkhornGibbs(), SinkhornStabilized())
            epsscaling_alg = SinkhornEpsilonScaling(alg)
            ForwardDiff.gradient(zeros(N)) do xs
                return sinkhorn2(μ, softmax(xs), C, ε, epsscaling_alg)
            end
            ForwardDiff.gradient(zeros(M)) do xs
                return sinkhorn2(softmax(xs), ν, C, ε, epsscaling_alg)
            end
            ForwardDiff.gradient(zeros(M + N)) do xs
                return sinkhorn2(
                    softmax(xs[1:M]), softmax(xs[(M + 1):end]), C, ε, epsscaling_alg
                )
            end
        end
    end

    @testset "deprecations" begin
        alg = SinkhornEpsilonScaling(
            SinkhornStabilized(; absorb_tol=10); factor=0.7, steps=2
        )
        γ = sinkhorn(μ, ν, C, ε, alg)
        @test (@test_deprecated sinkhorn_stabilized_epsscaling(
            μ, ν, C, ε; absorb_tol=10, lambda=0.7, k=2
        )) == γ
        @test (@test_deprecated sinkhorn_stabilized_epsscaling(
            μ, ν, C, ε; absorb_tol=10, scaling_factor=0.7, scaling_steps=2
        )) == γ
    end
end
