using OptimalTransport
import OptimalTransport.Dual: Dual
import MLDatasets
using StatsBase
using Plots
using LogExpFunctions
import NNlib
using LinearAlgebra
using Distances
using Base.Iterators

Plots.gr()

function simplex_norm!(x; dims = 1)
    x .= x ./ sum(x; dims = dims)
end

# factor = 2
# sizex, sizey = 28, 28
# Σ = hcat([sum(I(sizex)[:, i:i+factor-1]; dims = 2) for i = 1:factor:sizex]...)
# sizex, sizey = sizex÷factor, sizey÷factor
# 
# N = 64
# x, y = MLDatasets.MNIST.traindata(Float64, sample(1:60_000, N, replace = false))
# x = permutedims(x, (2, 1, 3))
# x = cat([Σ' * x[:, :, i] * Σ for i = 1:N]...; dims = 3)
# 
# # Preview 
# 
# M = 8
# for i = 1:M
#     PyPlot.subplot(2, M//2, i)
#     PyPlot.imshow(x[:, :, i])
# end
# 
# X = simplex_norm!(reshape(x, (sizex*sizey, :)))

using Distributions

f(x, μ, σ) = exp.(-(x .- μ).^2)
coord = range(-12, 12; length = 100)
N = 64
σ = 1
X = hcat([rand()*f(coord, σ*randn() + 6, 1) + rand()*f(coord, σ*randn(), 1) + rand()*f(coord, σ*randn() - 6, 1) for _ in 1:N]...)
X = simplex_norm!(X)
plot(coord, X[:, 1:3])

# Pick rank

# coord = reshape(collect(product(1:sizex, 1:sizey)), :)
C = pairwise(SqEuclidean(), coord)
C = C / mean(C)
ε = 0.05
ρ1, ρ2 = (5e-2, 5e-2)
k = 3
K = exp.(-C/ε)

D = rand(size(X, 1), k) # dictionary 
simplex_norm!(D; dims = 1) # norm columnwise
Λ = rand(k, size(X, 2)) # weights
simplex_norm!(Λ; dims = 1) # norm rowwise

function E_star(x; dims = 1) 
    return logsumexp(x; dims = dims)
end

function dual_obj_weights(X, K, ε, D, G, ρ1)
    sum(Dual.ot_entropic_semidual(X, G, ε, K)) + ρ1 * sum(E_star(-D' * G / ρ1))
end

function dual_obj_dict(X, K, ε, Λ, G, ρ2)
    sum(Dual.ot_entropic_semidual(X, G, ε, K)) + ρ2 * sum(E_star(-G * Λ' / ρ2))
end

function getprimal_weights(D, G, ρ1)
    return NNlib.softmax(-D'*G / ρ1; dims = 1)
end

function getprimal_dict(Λ, G, ρ2)
    return NNlib.softmax(-G*Λ' / ρ2; dims = 1)
end

using Optim

function solve_weights(X, K, ε, D, ρ1; alg = GradientDescent(), options = Optim.Options(; iterations = 50, g_tol = 1e-6, show_trace = true))
    opt = optimize(g -> dual_obj_weights(X, K, ε, D, g, ρ1),
             zero.(X),
             alg,
             options,
             autodiff=:forward)
    return getprimal_weights(D, Optim.minimizer(opt), ρ1)
end

function solve_dict(X, K, ε, Λ, ρ2; alg = GradientDescent(), options = Optim.Options(; iterations = 50, g_tol = 1e-6, show_trace = true))
    opt = optimize(g -> dual_obj_dict(X, K, ε, Λ, g, ρ2),
             zero.(X),
             alg,
             options,
             autodiff=:forward)
    return getprimal_dict(Λ, Optim.minimizer(opt), ρ2)
end

import LineSearches

n_iter = 5
for iter in 1:n_iter
    D = solve_dict(X, K, ε, Λ, ρ2; alg = LBFGS(linesearch = LineSearches.BackTracking()))
    Λ = solve_weights(X, K, ε, D, ρ1; alg = LBFGS(linesearch = LineSearches.BackTracking()))
end

plot(coord, D)
