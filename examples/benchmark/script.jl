using OptimalTransport
using PythonOT: PythonOT
using LinearAlgebra
using Random
using Tulip, Clp
using BenchmarkTools

Random.seed!(0)

N = 100
μ = fill(1 / N, N)
ν = fill(1 / N, N)
C = rand(N, N)

# Exact 
@benchmark begin
    emd2(μ, ν, C, Tulip.Optimizer())
end

@benchmark begin
    PythonOT.emd2(μ, ν, C)
end

# Entropy-regularised
N = 500
μ = normalize(rand(N), 1)
ν = normalize(rand(N), 1)
C = rand(N, N)
ε = 0.001 * mean(C)

@benchmark begin
    sinkhorn2(μ, ν, C, ε)
end

@benchmark begin
    PythonOT.sinkhorn2(μ, ν, C, ε)
end

# Quadratic
ε = 0.5 * mean(C)

ENV["JULIA_DEBUG"] = "OptimalTransport"
quadreg(μ, ν, C, ε)
