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
    ::Type{T}, ::SinkhornGibbs, size2::Tuple, μ::AbstractMatrix, C::AbstractMatrix, ε::Real
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

# Sinkhorn algorithm steps (see solve!)
prestep!(::SinkhornBarycenterSolver{SinkhornGibbs}, ::Int) = nothing

function init_step!(solver::SinkhornBarycenterSolver{SinkhornGibbs})
    return A_batched_mul_B!(solver.cache.Kv, solver.cache.K, solver.cache.v)
end

function step!(solver::SinkhornBarycenterSolver{SinkhornGibbs}, iter::Int)
    μ = solver.source
    w = solver.w
    cache = solver.cache
    u = cache.u
    v = cache.v
    Kv = cache.Kv
    K = cache.K
    a = cache.a

    a .= prod(Kv' .^ w; dims=1)'  # TODO: optimise 
    u .= a ./ Kv
    At_batched_mul_B!(v, K, u)
    v .= μ ./ v
    return A_batched_mul_B!(Kv, K, v)
end

function check_convergence(solver::SinkhornBarycenterSolver{SinkhornGibbs})
    return OptimalTransport.check_convergence(
        solver.cache.a,
        solver.cache.u,
        solver.cache.Kv,
        solver.convergence_cache,
        solver.atol,
        solver.rtol,
    )
end

function solution(solver::SinkhornBarycenterSolver{SinkhornGibbs})
    cache = solver.cache
    return cache.u[:, 1] .* cache.Kv[:, 1]
end
