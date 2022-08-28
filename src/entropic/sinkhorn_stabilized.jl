# algorithm

struct SinkhornStabilized{T<:Real} <: Sinkhorn
    absorb_tol::T
end

"""
    SinkhornStabilized(; absorb_tol::Real=1_000)

Construct a log-domain stabilized Sinkhorn algorithm with absorption tolerance `absorb_tol`
for solving an entropically regularized optimal transport problem.

# References

Schmitzer, B. (2019).
[Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems](https://doi.org/10.1137/16m1106018).
SIAM Journal on Scientific Computing, 41(3), A1443–A1481.
"""
function SinkhornStabilized(; absorb_tol::Real=1_000)
    absorb_tol > zero(absorb_tol) ||
        throw(ArgumentError("absorption tolerance `absorb_tol` must be positive"))
    return SinkhornStabilized(absorb_tol)
end

function Base.show(io::IO, alg::SinkhornStabilized)
    return print(
        io,
        "Log-domain stabilized Sinkhorn algorithm (absorption tolerance = ",
        alg.absorb_tol,
        ")",
    )
end

# caches

struct SinkhornStabilizedCache{U,V,KT}
    u::U
    v::V
    alpha::U
    beta::V
    K::KT
    Kv::U
end

function build_cache(
    ::Type{T},
    ::SinkhornStabilized,
    size2::Tuple,
    μ::AbstractVecOrMat,
    ν::AbstractVecOrMat,
    C::AbstractMatrix,
    ε::Real,
) where {T}
    # compute Gibbs kernel (has to be mutable for ε-scaling algorithm)
    K = similar(C, T, size(C)..., size2...)
    @. K = exp(-C / ε)

    # create and initialize dual potentials
    u = similar(μ, T, size(μ, 1), size2...)
    v = similar(ν, T, size(ν, 1), size2...)
    α = similar(u)
    β = similar(v)
    fill!(u, one(T))
    fill!(v, one(T))
    fill!(α, zero(T))
    fill!(β, zero(T))

    # cache for next iterate of `u`
    Kv = similar(u)

    return SinkhornStabilizedCache(u, v, α, β, K, Kv)
end

# interface

function prestep!(solver::SinkhornSolver{<:SinkhornStabilized}, iter::Int)
    # unpack
    absorb_tol = solver.alg.absorb_tol
    cache = solver.cache
    u = cache.u
    v = cache.v

    # absorption step
    if maximum(abs, u) > absorb_tol || maximum(abs, v) > absorb_tol
        @debug string(solver.alg) *
            " (" *
            string(iter) *
            "/" *
            string(maxiter) *
            ": absorbing `u` and `v` into `α` and `β`"

        # absorb `u` and `v` into `α` and `β`
        absorb!(solver)

        # re-compute cache for next iterate of `u`
        A_batched_mul_B!(cache.Kv, cache.K, v)
    end

    return nothing
end

# absorption step
function absorb!(solver::SinkhornSolver{<:SinkhornStabilized})
    # unpack
    ε = solver.eps
    cache = solver.cache
    u = cache.u
    v = cache.v
    α = cache.alpha
    β = cache.beta

    # update dual potentials
    α .+= ε .* log.(u)
    β .+= ε .* log.(v)

    # reset `α` and `β`
    fill!(u, one(eltype(u)))
    fill!(v, one(eltype(v)))

    # update kernel
    update_K!(solver)

    return nothing
end

# update kernel
function update_K!(solver::SinkhornSolver{<:SinkhornStabilized})
    cache = solver.cache
    α = add_singleton(cache.alpha, Val(2))
    β = add_singleton(cache.beta, Val(1))
    @. cache.K = exp(-(solver.C - α - β) / solver.eps)
    return nothing
end

# obtain plan
function sinkhorn_plan(solver::SinkhornSolver{<:SinkhornStabilized})
    absorb!(solver)
    return copy(solver.cache.K)
end

# deprecations

"""
    sinkhorn_stabilized(μ, ν, C, ε; absorb_tol=1_000, kwargs...)

This method is deprecated, please use
```julia
sinkhorn(
    μ, ν, C, ε, SinkhornStabilized(; absorb_tol=absorb_tol); kwargs...
)
```
instead.

See also: [`sinkhorn`](@ref), [`SinkhornStabilized`](@ref)
"""
function sinkhorn_stabilized(
    μ, ν, C, ε; tol=nothing, atol=tol, return_duals=nothing, absorb_tol=1_000, kwargs...
)
    # deprecation warnings
    if tol !== nothing
        Base.depwarn(
            "keyword argument `tol` is deprecated, please use `atol` and `rtol`",
            :sinkhorn_stabilized,
        )
    end
    if return_duals !== nothing
        Base.depwarn(
            "keyword argument `return_duals` is deprecated and will be removed",
            :sinkhorn_stabilized,
        )
    end

    Base.depwarn(
        "`sinkhorn_stabilized` is deprecated, " *
        "please use `sinkhorn` with `SinkhornStabilized` instead",
        :sinkhorn_stabilized_epsscaling,
    )

    # construct algorithm
    algorithm = SinkhornStabilized(; absorb_tol=absorb_tol)

    # return "dual potentials" (in log space and scaled by 1/ε) if requested
    if return_duals !== nothing && return_duals
        solver = build_solver(μ, ν, C, ε, algorithm; atol=atol, kwargs...)
        solve!(solver)
        absorb!(solver)
        cache = solver.cache
        return cache.alpha, cache.beta
    end

    return sinkhorn(μ, ν, C, ε, algorithm; atol=atol, kwargs...)
end
