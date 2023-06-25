using Revise
using Pkg
Pkg.activate(".")
using OptimalTransport
using LinearAlgebra
using SparseArrays
using Distances
using StatsBase
using NNlib
using LazyArrays
using IterativeSolvers

N = 10_000
d = 250
μ_spt = randn(N, d)
X = μ_spt
μ = ones(N)
C = sum(X.^2 ; dims = 2)/2 .+ sum(X.^2 ; dims = 2)'/2 - X * X'
Cmean = mean(C)
C[diagind(C)] .= Inf
ENV["JULIA_DEBUG"] = "OptimalTransport"

ε = 1.0
π = sparse(quadreg(μ, C / Cmean, ε, OptimalTransport.SymmetricQuadraticOTNewton(); maxiter = 25))

nnz(π) / N^2

using NearestNeighborDescent
using NearestNeighbors

function knn_adj(X, k)
    # indices, _ = knn_matrices(nndescent(X, k, Euclidean())); 
    indices, _ = knn(KDTree(X), X, k);
    A = spzeros(size(X, 2), size(X, 2));
    @inbounds for i = 1:size(A, 1)
        A[i, i] = 1
        @inbounds for j in indices[i]
            A[i, j] = 1
        end
    end
    return A
end

S = knn_adj(X', 50);
S += sign.(mean([randPermutation(N) for _ in 1:50])); 
S = sign.(S); 
S[diagind(S)] .= 0;
dropzeros!(S);

nnz(S) / N^2

ε = 1
alg = OptimalTransport.SymmetricQuadraticOTNewtonAS(S)
γ = quadreg(μ, C / Cmean, ε, alg; maxiter = 100)

sum(γ; dims = 2)
sum(γ; dims = 1)

diag(γ)

norm(π - γ, 1)

D = sum(X.^2 ; dims = 2)/2
alg = OptimalTransport.SymmetricQuadraticOTNewtonAS(S)
# form partial cost matrix
I, J, V = findnz(S)
C = D[I] + D[J]; @time @inbounds for k = 1:length(V) C[k] += -dot(X[I[k], :], X[J[k], :]) end
C_sp = sparse(I, J, relu.(C))
solver = OptimalTransport.build_solver(μ, C_sp, ε, alg; maxiter = 50)
OptimalTransport.solve!(solver);
# form full cost matrix
Cfull = sum(X.^2 ; dims = 2)/2 .+ sum(X.^2 ; dims = 2)'/2 - X * X';
Cfull[diagind(Cfull)] .= Inf;
plan = OptimalTransport.plan(solver, Cfull)
S = max.(S, sign.(plan))

norm(sum(plan; dims = 1) .- 1)
norm(sum(plan; dims = 2) .- 1)

nnz(plan) / N^2

plot(heatmap(Array(plan)), heatmap(Array(π)))

"""
D = sum(X.^2 ; dims = 2)/2
Csp = D[I] + D[J]; @time @inbounds for k = 1:length(V) Csp[k] += -dot(X[I[k], :], X[J[k], :]) end
C = sum(X.^2 ; dims = 2)/2 .+ sum(X.^2 ; dims = 2)'/2 - X * X'
C[diagind(C)] .= Inf
## start 
u = zeros(N)

function Φ(u, μ, C, ε, I, J)
    return norm(NNlib.relu.(u[I] + u[J] - C))^2 / 2 - 2*ε * dot(μ, u) 
end

armijo_max = 25
δ = 1e-5
for it = 1:10
    @info it
    I, J, V = findnz(S)
    Csp = similar(X, length(V));
    @inbounds for k = 1:length(V) Csp[k] = D[I[k]] + D[J[k]] - dot(X[I[k], :], X[J[k], :]) end
    γ = u[I] + u[J] - Csp
    σ = similar(Csp)
    @. σ = γ ≥ 0
    @. γ = relu(γ) / ε
    σ_sp = sparse(I, J, σ)
    γ_sp = sparse(I, J, γ)
    G = σ_sp + Diagonal(vec(sum(σ_sp; dims = 2)) .+ δ)
    b = -ε*(vec(sum(γ_sp; dims = 2)) .- 1)

    δu = similar(u)
    fill!(δu, 0)
    cg!(δu, G, b)

    d = ε*dot(γ, δu[I] + δu[J]) - 2ε*dot(δu, μ)
    @info d
    t = 1
    Φ0 = Φ(u, μ, Csp, ε, I, J)
    armijo_counter = 0
    θ = alg.θ
    κ = alg.κ
    while (armijo_counter < armijo_max) &&
        (Φ(u + t * δu, μ, Csp, ε, I, J) ≥ Φ0 + t * θ * d)
        t = κ * t
        armijo_counter += 1
    end
    u .= u + t * δu
end
γ = relu.(u .+ u' - C) / ε
S = max.(S, sign.(γ))
norm(γ - π, 1) / norm(γ, 1)

sum(γ; dims = 1)
sum(γ; dims = 2)

sparse(γ)


using Plots
plot(heatmap(Array(γ)), heatmap(Array(π)))
"""
