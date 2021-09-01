using LinearAlgebra
using LogExpFunctions
using StatsBase
import OptimalTransport.add_singleton

function logKexp(logK::AbstractMatrix, x::AbstractVector)
    return logsumexp(logK .+ add_singleton(x, Val(1)); dims = 2)
end

function logKexp(logK::AbstractMatrix, x::AbstractMatrix; dims)
    return logsumexp(logK + x; dims = dims)
end

"""
    ot_entropic_semidual(μ, v, eps, K; stabilized = false)

Computes the semidual (in the second argument) of the entropic optimal transport loss, with source marginal `μ`, regularization parameter `ε`, and Gibbs kernel `K`.
That is, if
```math
    \\operatorname{OT}_{\\varepsilon}(\\mu, \\nu) = \\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle + \\varepsilon \\Omega(\\gamma)
```
with ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{ij} \\log \\gamma_{ij}``, then the semidual in the second argument ν is
```math

    \\operatorname{OT}_{\\varepsilon}^*(\\mu, v) = \\sup_{\\nu} \\langle v, \\nu \\rangle - \\operatorname{OT}_{\\varepsilon}(\\mu, \\nu). 
```
"""
function ot_entropic_semidual(μ, v, eps, K; stabilized = false)
    if stabilized
        return eps*(-sum(xlogx.(μ) .- μ) + dot(μ, logKexp(K, v/eps)))
    else
        return eps*(-sum(xlogx.(μ) .- μ) + dot(μ, log.(K*exp.(v/eps))))
    end
end

function getprimal_ot_entropic_semidual(μ, v, eps, K)
    return Diagonal(exp.(v/eps)) * K' * (μ ./ (K * exp.(v/eps)))
end

function ot_entropic_dual(u, v, eps, K; stabilized = false)
    # (μ, ν) → min_{γ ∈ Π(μ, ν)} ε H(γ | K)
    # has Legendre transform
    # (u, v) → ε log < exp(u/ε), K exp(v/ε) >
    if stabilized
        # K is actually logK
        return eps * logKexp(K, add_singleton(u/eps, Val(2)) .+ add_singleton(v/eps, Val(1)); dims = :)
    else
        return eps * log(dot(exp.(u/eps), K * exp.(v/eps)))
    end
end
