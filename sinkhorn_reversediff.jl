using OptimalTransport
using ReverseDiff
using ForwardDiff
using LogExpFunctions
import NNlib
using LinearAlgebra

# pair of histograms
N = 100 
C = rand(N, N)
ε = 0.05

μ = NNlib.softmax(zeros(N,); dims = 1)
ν = NNlib.softmax(zeros(N,); dims = 1)

∇_r = ReverseDiff.gradient(zero(ν)) do x
    sinkhorn2(μ, NNlib.softmax(x; dims = 1), C, ε; regularization = true)
end

∇_f = ForwardDiff.gradient(zero(ν)) do x
    sinkhorn2(μ, NNlib.softmax(x; dims = 1), C, ε; regularization = true)
end

∇_r ≈ ∇_f 

# batch 
M = 25
μ = NNlib.softmax(zeros(N, M); dims = 1)
ν = NNlib.softmax(zeros(N, M); dims = 1)

∇_r = ReverseDiff.gradient(zero(ν)) do x
    sum(sinkhorn2(μ, NNlib.softmax(x; dims = 1), C, ε; regularization = true))
end

∇_f = ForwardDiff.gradient(zero(ν)) do x
    sum(sinkhorn2(μ, NNlib.softmax(x; dims = 1), C, ε; regularization = true))
end

∇_r ≈ ∇_r 
