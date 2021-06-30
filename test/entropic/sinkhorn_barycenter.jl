using OptimalTransport

using Distances
using PythonOT: PythonOT

using LinearAlgebra
using Random
using Test

const POT = PythonOT

Random.seed!(100)

@testset "sinkhorn_barycenter.jl" begin
    @testset "example" begin
        # set up support
        support = range(-1; stop=1, length=250)
        μ1 = normalize!(exp.(-(support .+ 0.5) .^ 2 ./ 0.1^2), 1)
        μ2 = normalize!(exp.(-(support .- 0.5) .^ 2 ./ 0.1^2), 1)
        μ_all = hcat(μ1, μ2)

        # create cost matrix
        C = pairwise(SqEuclidean(), support'; dims=2)

        # compute Sinkhorn barycenter
        eps = 0.01
        μ_interp = sinkhorn_barycenter(μ_all, C, eps, [0.5, 0.5])

        # compare with POT
        # need to use a larger tolerance here because of a quirk with the POT solver
        μ_interp_pot = POT.barycenter(μ_all, C, eps; weights=[0.5, 0.5], stopThr=1e-9)
        @test μ_interp ≈ μ_interp_pot rtol = 1e-6
    end
end
