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

function ot_entropic_semidual(μ, v, eps, K)
    return eps*(-sum(xlogx.(μ) .- μ) + dot(μ, logKexp(K, v/eps)))
end

function getprimal_ot_entropic_semidual(μ, v, eps, K)
    return Diagonal(exp.(v/eps)) * K' * (μ ./ (K * exp.(v/eps)))
end

function ot_entropic_dual(u, v, eps, K; stabilized = true)
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
