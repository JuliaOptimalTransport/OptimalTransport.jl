struct SinkhornBarycenterGibbs <: Sinkhorn end

# solver cache
function build_cache(
    ::Type{T},
    ::SinkhornBarycenterGibbs,
    size2::Tuple,
    μ::AbstractMatrix,
    C::AbstractMatrix,
    ε::Real,
) where {T}
    return build_cache(T, SinkhornGibbs(), size2, μ, μ, C, ε)
end

prestep!(::SinkhornBarycenterSolver{SinkhornBarycenterGibbs}, ::Int) = nothing

function solution(solver::SinkhornBarycenterSolver{SinkhornBarycenterGibbs})
    cache = solver.cache
    return cache.u[:, 1] .* cache.Kv[:, 1]
end
