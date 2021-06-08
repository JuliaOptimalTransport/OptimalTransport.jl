# algorithm

"""
    SinkhornGibbs

Vanilla Sinkhorn algorithm.
"""
struct SinkhornGibbs <: Sinkhorn end

Base.show(io::IO, ::SinkhornGibbs) = print(io, "Sinkhorn algorithm")

# cache

struct SinkhornGibbsCache{U,V,KT}
    u::U
    v::V
    K::KT
    Kv::U
end

function build_cache(
    ::Type{T},
    ::SinkhornGibbs,
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
    fill!(u, one(T))
    fill!(v, one(T))

    # cache for next iterate of `u`
    Kv = similar(u)

    return SinkhornGibbsCache(u, v, K, Kv)
end

# use `SinkhornGibbs` as default algorithm
# TODO: remove deprecations
function sinkhorn(
    μ,
    ν,
    C,
    ε;
    tol=nothing,
    atol=tol,
    check_marginal_step=nothing,
    check_convergence=check_marginal_step,
    kwargs...,
)
    if tol !== nothing
        Base.depwarn(
            "keyword argument `tol` is deprecated, please use `atol` and `rtol`", :sinkhorn
        )
    end
    if check_marginal_step !== nothing
        Base.depwarn(
            "keyword argument `check_marginal_step` is deprecated, please use `check_convergence`",
            :sinkhorn,
        )
    end

    _check_convergence = check_convergence === nothing ? 10 : check_convergence

    return sinkhorn(
        μ,
        ν,
        C,
        ε,
        SinkhornGibbs();
        atol=atol,
        check_convergence=_check_convergence,
        kwargs...,
    )
end
function sinkhorn2(
    μ,
    ν,
    C,
    ε;
    tol=nothing,
    atol=tol,
    check_marginal_step=nothing,
    check_convergence=check_marginal_step,
    kwargs...,
)
    if tol !== nothing
        Base.depwarn(
            "keyword argument `tol` is deprecated, please use `atol` and `rtol`", :sinkhorn2
        )
    end
    if check_marginal_step !== nothing
        Base.depwarn(
            "keyword argument `check_marginal_step` is deprecated, please use `check_convergence`",
            :sinkhorn2,
        )
    end

    _check_convergence = check_convergence === nothing ? 10 : check_convergence

    return sinkhorn2(
        μ,
        ν,
        C,
        ε,
        SinkhornGibbs();
        atol=atol,
        check_convergence=_check_convergence,
        kwargs...,
    )
end

# interface

prestep!(::SinkhornSolver{SinkhornGibbs}, ::Int) = nothing

function plan(solver::SinkhornSolver{SinkhornGibbs})
    cache = solver.cache
    γ = cache.K .* add_singleton(cache.u, Val(2)) .* add_singleton(cache.v, Val(1))
    return γ
end
