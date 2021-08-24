using OptimalTransport

using Distances
using PythonOT: PythonOT

using Random
using Test

const POT = PythonOT

Random.seed!(100)

@testset "quadratic.jl" begin
    @testset "quadreg" begin
        M = 250
        N = 200

        # create two uniform histograms
        μ = fill(1 / M, M)
        ν = fill(1 / N, N)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.25
        γ = quadreg(μ, ν, C, eps, QuadraticOTNewton())

        γ_pot = POT.Smooth.smooth_ot_dual(μ, ν, C, eps; stopThr=1e-9)

        # need to use a larger tolerance here because of a quirk with the POT solver
        @test γ ≈ γ_pot rtol = 1e-1
    end
end
