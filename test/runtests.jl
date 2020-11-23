using OptimalTransport
using CUDA
using Distances

using LinearAlgebra
using Random
using Test

Random.seed!(100)
@testset "Earth-Movers Distances transport" begin
    @testset "Clp and Tulip" begin
        N = 200
        M = 250
        μ = rand(N)
        ν = rand(M)
        μ = μ/sum(μ)
        ν = ν/sum(ν)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims = 2)
        cost = emd2(μ,ν,C,"IPM")
        cost_clp = emd2(μ,ν,C,"Simplex")

        μ = vcat(μ,zeros(M-N))       # Requires dims(mu) == dims(nu)
        C = hcat(C,zeros(M,M-N))     # Add columns so that C is MxM
        pot_cost = pot_emd2(μ, ν, C) # Requires dims(mu) == dims(nu)
        @test cost ≈ pot_cost atol=1e-7
        @test cost_clp ≈ pot_cost atol=1e-7
    end
    @testset "Native implementation" begin
        N = 20
        M = 25
        μ = rand(N)
        ν = rand(M)
        μ = μ/sum(μ)
        ν = ν/sum(ν)

        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims = 2)
        cost = emd2(μ,ν,C,"Native")
        # create random cost matrix
        μ = vcat(μ,zeros(M-N))       # Requires dims(mu) == dims(nu)
        C = hcat(C,zeros(M,M-N))     # Add columns so that C is MxM
        pot_cost = pot_emd2(μ, ν, C) # Requires dims(mu) == dims(nu)
        @test cost ≈ pot_cost atol=1e-7
    end
end


@testset "entropically regularized transport" begin
    M = 250
    N = 200

    @testset "example" begin
        # create two uniform histograms
        μ = fill(1/M, M)
        ν = fill(1/N, N)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims = 2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.01
        γ = sinkhorn(μ, ν, C, eps)
        γ_pot = OptimalTransport.pot_sinkhorn(μ, ν, C, eps)
        @test norm(γ - γ_pot, Inf) < 1e-9

        # compute optimal transport cost (Julia implementation + POT)
        c = sinkhorn2(μ, ν, C, eps)
        c_pot = OptimalTransport.pot_sinkhorn2(μ, ν, C, eps)
        @test c ≈ c_pot atol=1e-9
    end

    # different element type
    @testset "Float32" begin
        # create two uniform histograms
        μ = fill(Float32(1/M), M)
        ν = fill(Float32(1/N), N)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(Float32, 1, M), rand(Float32, 1, N); dims = 2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.01f0
        γ = sinkhorn(μ, ν, C, eps)
        @test eltype(γ) === Float32

        γ_pot = OptimalTransport.pot_sinkhorn(μ, ν, C, eps)
        @test eltype(γ_pot) === Float64 # POT does not respect input type
        @test norm(γ - γ_pot, Inf) < Base.eps(Float32)

        # compute optimal transport cost (Julia implementation + POT)
        c = sinkhorn2(μ, ν, C, eps)
        @test c isa Float32

        c_pot = OptimalTransport.pot_sinkhorn2(μ, ν, C, eps)
        @test c_pot isa Float64 # POT does not respect input types
        @test c ≈ c_pot atol=Base.eps(Float32)
    end

    # computation on the GPU
    if CUDA.functional()
        @testset "CUDA" begin
            # create two uniform histograms
            μ = CUDA.fill(Float32(1/M), M)
            ν = CUDA.fill(Float32(1/N), N)

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
        μ = fill(1/N, M)
        ν = fill(1/N, N)
        
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims = 2)

        eps = 0.01
        lambda = 1
        γ = sinkhorn_unbalanced(μ, ν, C, lambda, lambda, eps)
        γ_pot = pot_sinkhorn_unbalanced(μ, ν, C, eps, lambda)

        # compute optimal transport map
        @test norm(γ - γ_pot, Inf) < 1e-9

        c = sinkhorn_unbalanced2(μ, ν, C, lambda, lambda, eps)
        c_pot = pot_sinkhorn_unbalanced2(μ, ν, C, eps, lambda)

        @test c ≈ c_pot atol=1e-9
    end
end


@testset "stabilized sinkhorn" begin
    M = 250
    N = 200

    @testset "example" begin
        # create two uniform histograms
        μ = fill(1/M, M)
        ν = fill(1/N, N)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims = 2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.01
        γ = sinkhorn_stabilized(μ, ν, C, eps)
        γ_pot = OptimalTransport.pot_sinkhorn(μ, ν, C, eps, method = "sinkhorn_stabilized")
        @test norm(γ - γ_pot, Inf) < 1e-9
    end
end
