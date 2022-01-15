# Greenkhorn is a greedy version of the Sinkhorn algorithm
# This method is from https://arxiv.org/pdf/1705.09634.pdf
# Code is based on implementation from package POT


"""
    Greenkhorn()

Greenkhorn is a greedy version of the Sinkhorn algorithm.
"""
struct Greenkhorn <: Sinkhorn end

struct GreenkhornCache{U,V,KT}
    u::U
    v::V
    K::KT
    Kv::U
    G::KT
    du::U
    dv::V
end

Base.show(io::IO, ::Greenkhorn) = print(io, "Greenkhorn algorithm")

function build_cache(
    ::Type{T},
    ::Greenkhorn,
    size2::Tuple,
    μ::AbstractVecOrMat,
    ν::AbstractVecOrMat,
    C::AbstractMatrix,
    ε::Real,
) where {T}
    # compute Gibbs kernel (has to be mutable for ε-scaling algorithm)
    K = similar(C, T)
    @. K = exp(-C / ε)

    # create and initialize dual potentials
    u = similar(μ, T, size(μ, 1), size2...)
    v = similar(ν, T, size(ν, 1), size2...)
    fill!(u, one(T)/size(μ, 1))
    fill!(v, one(T)/size(ν, 1))

    # G = sinkhorn_plan(u, v, K)
    G = diagm(u) * K * diagm(v) 

    Kv = similar(u)

    # improve this!
    # du = similar(u)
    # fill!(du, 1.0)
    # dv = similar(v)
    du = sum(G', dims=1)[:] - μ
    dv = sum(G', dims=2)[:] - ν

    return GreenkhornCache(u, v, K, Kv, G, du, dv)
end

prestep!(::SinkhornSolver{Greenkhorn}, ::Int) = nothing

function init_step!(solver::SinkhornSolver{<:Greenkhorn})
    return A_batched_mul_B!(solver.cache.Kv, solver.cache.K, solver.cache.v)
end

function step!(solver::SinkhornSolver{<:Greenkhorn}, iter::Int)
    μ = solver.source
    ν = solver.target
    cache = solver.cache
    u = cache.u
    v = cache.v
    K = cache.K
    G = cache.G
    Δμ= cache.du
    Δν= cache.dv

    ρμ = abs.(Δμ + μ .* log.(μ ./ sum(G', dims=1)[:]))
    ρν = abs.(Δν + ν .* log.(ν ./ sum(G', dims=2)[:]))
    
    i₁ = argmax(ρμ)
    i₂ = argmax(ρν)

    if ρμ[i₁]> ρν[i₂]
        old_u = u[i₁]
        u[i₁] = μ[i₁]/ (K[i₁,:] ⋅ v)
        Δ = u[i₁] - old_u
        G[i₁, :] = u[i₁] * K[i₁,:] .* v
        Δμ[i₁] = u[i₁] * (K[i₁,:] ⋅ v) - μ[i₁]
        Δν = Δν .+  Δ .* K[i₁,:] .* v
    else
        old_v = v[i₂]
        v[i₂] = ν[i₂]/ (K[:,i₂] ⋅ u)
        Δ = v[i₂] - old_v
        G[:, i₂] = v[i₂] * K[:,i₂] .* u
        Δν[i₂] = v[i₂] * (K[:,i₂] ⋅ u) - ν[i₂]
        Δμ = Δμ .+  Δ .* K[:,i₂] .* u
    end
    cache.du .= Δμ
    cache.dv .= Δν
    A_batched_mul_B!(solver.cache.Kv, K, v)
end

function sinkhorn_plan(solver::SinkhornSolver{Greenkhorn})
    cache = solver.cache
    # println("OK")
    # println('K',cache.K)
    # println('u',cache.u)
    # println('v',cache.v)
    # 
    # println("μ",solver.source)
    # println("ν",solver.target)
    # return sinkhorn_plan(cache.u, cache.v, cache.K)
    return cache.G
end

