using OptimalTransport

using Distances
using PythonOT: PythonOT

using LinearAlgebra
using Random
using Test

const POT = PythonOT

Random.seed!(100)

@testset "sinkhorn_barycenter.jl" begin
    # set up support
    support = range(-1; stop=1, length=500)
    N = 10
    μ = hcat([normalize!(exp.(-(support .+ rand()) .^ 2 ./ 0.1^2), 1) for _ in 1:N]...)

    # create cost matrix
    C = pairwise(SqEuclidean(), support'; dims=2)

    # regularisation parameter
    ε = 0.05

    # weights 
    w = ones(N) / N

    @testset "example" begin
        α = sinkhorn_barycenter(μ, C, ε, w, SinkhornGibbs())

        # compare with POT
        # need to use a larger tolerance here because of a quirk with the POT solver
        α_pot = POT.barycenter(μ, C, ε; weights=w, stopThr=1e-6)
        @test α ≈ α_pot rtol = 1e-6
    end

    # different element type
    @testset "Float32" begin
        μ32 = map(Float32, μ)
        ε32 = map(Float32, ε)
        C32 = map(Float32, C)
        w32 = map(Float32, w)
        α = sinkhorn_barycenter(μ32, C32, ε32, w32, SinkhornGibbs())

        α_pot = POT.barycenter(μ32, C32, ε32; weights=w32, stopThr=1e-6)
        @test α ≈ α_pot rtol = 1e-5
    end
end
