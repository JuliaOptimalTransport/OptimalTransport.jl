# algorithm

struct SinkhornBarycenterGibbs <: Sinkhorn end

# cache

struct SinkhornBarycenterGibbsCache{U,V,KT,A}
    u::U
    v::V
    K::KT
    Kv::U
    a::A
end

# solver cache
function build_cache(
    ::Type{T},
    ::SinkhornBarycenterGibbs,
    size2::Tuple,
    μ::AbstractMatrix,
    C::AbstractMatrix,
    ε::Real,
) where {T}
    # compute Gibbs kernel (has to be mutable for ε-scaling algorithm)
    K = similar(C, T)
    @. K = exp(-C / ε)

    # create and initialize dual potentials
    u = similar(μ, T, size(μ, 1), size2...)
    v = similar(μ, T, size(μ, 1), size2...)
    a = similar(μ, T, size(μ, 1), 1)
    fill!(u, one(T))
    fill!(v, one(T))
    fill!(a, one(T))

    # cache for next iterate of `u`
    Kv = similar(u)

    return SinkhornBarycenterGibbsCache(u, v, K, Kv, a)
end

prestep!(::SinkhornBarycenterSolver{SinkhornBarycenterGibbs}, ::Int) = nothing

function solution(solver::SinkhornBarycenterSolver{SinkhornBarycenterGibbs})
    cache = solver.cache
    return cache.u[:, 1] .* cache.Kv[:, 1]
end
