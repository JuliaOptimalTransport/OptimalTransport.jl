using Revise
using Pkg
Pkg.activate(".")
using OptimalTransport
using LinearAlgebra
using SparseArrays
using Distances
using StatsBase

N = 500
d = 500
μ_spt = randn(N, d)
X = μ_spt
μ = ones(N)
C = sum(X.^2 ; dims = 2)/2 .+ sum(X.^2 ; dims = 2)'/2 - X * X'
C[diagind(C)] .= Inf

ε = 5.0
π = sparse(quadreg(μ, C, ε, OptimalTransport.SymmetricQuadraticOTNewton(); maxiter = 25))

using RandomMatrix

# KNN initialisation
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

S = knn_adj(X', 25)
S += sign.(mean([randPermutation(N) for _ in 1:25]))
S = sign.(S)
S[diagind(S)] .= 0
dropzeros!(S)

using NNlib
using LazyArrays
using IterativeSolvers

D = sum(X.^2 ; dims = 2)/2

## start 
u = zeros(N)

function Φ(u, μ, C, ε, S)
    return norm(NNlib.relu.((u .* S) .+ (u' .* S) .- C))^2 / 2 - 2*ε * dot(μ, u) 
end

armijo_max = 25
for it = 1:10
    @info it
    Csp = similar(S);
    I, J, V = findnz(Csp);
    for (i, j) in zip(I, J) Csp[i, j] = D[i] + D[j] - dot(X[i, :], X[j, :]) end

    γ = (u .* S) + (u' .* S) - Csp
    σ = similar(S)
    I, J, V = findnz(σ)
    for (i, j) in zip(I, J) σ[i, j] = (γ[i, j] ≥ 0) end
    dropzeros!(σ)
    γ = relu.(γ) / ε
    δ = 1e-5

    G = σ + Diagonal(vec(sum(σ; dims = 2)) .+ δ)
    b = -ε*(vec(sum(γ; dims = 2)) .- 1)

    δu = similar(u)
    fill!(δu, 0)
    cg!(δu, G, b)


    d = ε*sum(γ .* ((δu .* S) + (δu' .* S))) - 2ε*dot(δu, μ)
    t = 1
    Φ0 = Φ(u, μ, Csp, ε, S)
    armijo_counter = 0
    θ = alg.alg.θ
    κ = alg.alg.κ
    while (armijo_counter < armijo_max) &&
        (Φ(u + t * δu, μ, Csp, ε, S) ≥ Φ0 + t * θ * d)
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
