# algorithm

abstract type Sinkhorn end

# solver

struct SinkhornSolver{A<:Sinkhorn,M,N,CT,E<:Real,T<:Real,R<:Real,C1,C2}
    source::M
    target::N
    C::CT
    eps::E
    alg::A
    atol::T
    rtol::R
    maxiter::Int
    check_convergence::Int
    cache::C1
    convergence_cache::C2
end

function build_solver(
    μ::AbstractVecOrMat,
    ν::AbstractVecOrMat,
    C::AbstractMatrix,
    ε::Real,
    alg::Sinkhorn;
    atol=nothing,
    rtol=nothing,
    check_convergence=10,
    maxiter::Int=1_000,
)
    # check that source and target marginals have the correct size and are balanced
    checksize(μ, ν, C)
    size2 = checksize2(μ, ν)
    checkbalanced(μ, ν)

    # compute type
    T = float(Base.promote_eltype(μ, ν, one(eltype(C)) / ε))

    # build caches
    cache = build_cache(T, alg, size2, μ, ν, C, ε)
    convergence_cache = build_convergence_cache(T, size2, μ)

    # set tolerances
    _atol = atol === nothing ? 0 : atol
    _rtol = rtol === nothing ? (_atol > zero(_atol) ? zero(T) : sqrt(eps(T))) : rtol

    # create solver
    solver = SinkhornSolver(
        μ, ν, C, ε, alg, _atol, _rtol, maxiter, check_convergence, cache, convergence_cache
    )

    return solver
end

# convergence caches

struct SinkhornConvergenceCache{U,S<:Real}
    tmp::U
    norm_source::S
end

struct SinkhornBatchConvergenceCache{U,S,C}
    tmp::U
    tmp2::U
    norm_source::S
    norm_uKv::U
    norm_diff::U
    isconverged::C
end

function build_convergence_cache(::Type{T}, ::Tuple{}, μ::AbstractVector) where {T}
    tmp = similar(μ, T)
    norm_μ = sum(abs, μ)
    return SinkhornConvergenceCache(tmp, norm_μ)
end

function build_convergence_cache(
    ::Type{T}, size2::Tuple{Int}, μ::AbstractVecOrMat
) where {T}
    tmp = similar(μ, T, size(μ, 1), size2...)
    tmp2 = similar(tmp)
    norm_μ = μ isa AbstractVector ? sum(abs, μ) : sum(abs, μ; dims=1)
    norm_uKv = similar(tmp, 1, size2...)
    norm_diff = similar(tmp, 1, size2...)
    isconverged = similar(tmp, Bool, 1, size2...)
    return SinkhornBatchConvergenceCache(
        tmp, tmp2, norm_μ, norm_uKv, norm_diff, isconverged
    )
end

# Sinkhorn algorithm

function solve!(solver::SinkhornSolver)
    # unpack solver
    μ = solver.source
    ν = solver.target
    atol = solver.atol
    rtol = solver.rtol
    maxiter = solver.maxiter
    check_convergence = solver.check_convergence
    cache = solver.cache
    convergence_cache = solver.convergence_cache

    # unpack cache
    u = cache.u
    v = cache.v
    K = cache.K
    Kv = cache.Kv

    A_batched_mul_B!(Kv, K, v)

    isconverged = false
    to_check_step = check_convergence
    for iter in 1:maxiter
        # computations before the Sinkhorn iteration (e.g., absorption step)
        prestep!(solver, iter)

        # perform Sinkhorn iteration
        u .= μ ./ Kv
        At_batched_mul_B!(v, K, u)
        v .= ν ./ v
        A_batched_mul_B!(Kv, K, v)

        # check source marginal
        # always check convergence after the final iteration
        to_check_step -= 1
        if to_check_step == 0 || iter == maxiter
            # reset counter
            to_check_step = check_convergence

            isconverged, abserror = OptimalTransport.check_convergence(
                μ, u, Kv, convergence_cache, atol, rtol
            )
            @debug string(solver.alg) *
                   " (" *
                   string(iter) *
                   "/" *
                   string(maxiter) *
                   ": absolute error of source marginal = " *
                   string(maximum(abserror))

            if isconverged
                @debug "$(solver.alg) ($iter/$maxiter): converged"
                break
            end
        end
    end

    if !isconverged
        @warn "$(solver.alg) ($maxiter/$maxiter): not converged"
    end

    return nothing
end

# for single inputs
function check_convergence(
    μ::AbstractVector,
    u::AbstractVector,
    Kv::AbstractVector,
    cache::SinkhornConvergenceCache,
    atol::Real,
    rtol::Real,
)
    # unpack
    tmp = cache.tmp
    norm_μ = cache.norm_source

    # do not overwrite `Kv` but reuse it for computing `u` if not converged
    tmp .= u .* Kv
    norm_uKv = sum(abs, tmp)
    tmp .= abs.(μ .- tmp)
    norm_diff = sum(tmp)

    isconverged = norm_diff < max(atol, rtol * max(norm_μ, norm_uKv))

    return isconverged, norm_diff
end

# for batches
function check_convergence(
    μ::AbstractVecOrMat,
    u::AbstractMatrix,
    Kv::AbstractMatrix,
    cache::SinkhornBatchConvergenceCache,
    atol::Real,
    rtol::Real,
)
    # unpack
    tmp = cache.tmp
    tmp2 = cache.tmp2
    norm_μ = cache.norm_source
    norm_uKv = cache.norm_uKv
    norm_diff = cache.norm_diff
    isconverged = cache.isconverged

    # do not overwrite `Kv` but reuse it for computing `u` if not converged
    tmp .= u .* Kv
    tmp2 .= abs.(tmp)
    sum!(norm_uKv, tmp2)
    tmp .= abs.(μ .- tmp)
    sum!(norm_diff, tmp)

    # check stopping criterion
    @. isconverged = norm_diff < max(atol, rtol * max(norm_μ, norm_uKv))

    return all(isconverged), norm_diff
end

# API

"""
    sinkhorn(
        μ, ν, C, ε, alg=SinkhornGibbs();
        atol=0, rtol=atol > 0 ? 0 : √eps, check_convergence=10, maxiter=1_000,
    )

Compute the optimal transport plan for the entropically regularized optimal transport
problem with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, and entropic regularization parameter `ε`.

The optimal transport plan `γ` is of the same size as `C` and solves
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma),
```
where ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{i,j} \\log \\gamma_{i,j}`` is the entropic
regularization term.

Every `check_convergence` steps it is assessed if the algorithm is converged by checking if
the iterate of the transport plan `G` satisfies
```julia
isapprox(sum(G; dims=2), μ; atol=atol, rtol=rtol, norm=x -> norm(x, 1))
```
The default `rtol` depends on the types of `μ`, `ν`, and `C`. After `maxiter` iterations,
the computation is stopped.

Batch computations for multiple histograms with a common cost matrix `C` can be performed by
passing `μ` or `ν` as matrices whose columns correspond to histograms. It is required that
the number of source and target marginals is equal or that a single source or single target
marginal is provided (either as matrix or as vector). The optimal transport plans are
returned as three-dimensional array where `γ[:, :, i]` is the optimal transport plan for the
`i`th pair of source and target marginals.

See also: [`sinkhorn2`](@ref)
"""
function sinkhorn(μ, ν, C, ε, alg::Sinkhorn; kwargs...)
    # build solver
    solver = build_solver(μ, ν, C, ε, alg; kwargs...)

    # perform Sinkhorn algorithm
    solve!(solver)

    # compute optimal transport plan
    γ = plan(solver)

    return γ
end

"""
    sinkhorn2(
        μ, ν, C, ε, alg=SinkhornGibbs(); regularization=false, plan=nothing, kwargs...
    )

Solve the entropically regularized optimal transport problem with source and target
marginals `μ` and `ν`, cost matrix `C` of size `(length(μ), length(ν))`, and entropic
regularization parameter `ε`, and return the optimal cost.

A pre-computed optimal transport `plan` may be provided. The other keyword arguments
supported here are the same as those in the [`sinkhorn`](@ref) function.

!!! note
    As the `sinkhorn2` function in the Python Optimal Transport package, this function
    returns the optimal transport cost without the regularization term. The cost
    with the regularization term can be computed by setting `regularization=true`.

See also: [`sinkhorn`](@ref)
"""
function sinkhorn2(μ, ν, C, ε, alg::Sinkhorn; regularization=false, plan=nothing, kwargs...)
    γ = if plan === nothing
        sinkhorn(μ, ν, C, ε, alg; kwargs...)
    else
        # check dimensions
        checksize(μ, ν, C)
        size(plan) == size(C) || error(
            "optimal transport plan `plan` and cost matrix `C` must be of the same size",
        )
        plan
    end
    cost = if regularization
        dot_matwise(γ, C) .+
        ε * reshape(sum(LogExpFunctions.xlogx, γ; dims=(1, 2)), size(γ)[3:end])
    else
        dot_matwise(γ, C)
    end

    return cost
end
