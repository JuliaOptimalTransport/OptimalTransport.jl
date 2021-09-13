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

# Sinkhorn algorithm steps (see solve!)
function init_step!(solver::SinkhornSolver)
    return A_batched_mul_B!(solver.cache.Kv, solver.cache.K, solver.cache.v)
end

function step!(solver::SinkhornSolver, iter::Int)
    μ = solver.source
    ν = solver.target
    cache = solver.cache
    u = cache.u
    v = cache.v
    Kv = cache.Kv
    K = cache.K

    u .= μ ./ Kv
    At_batched_mul_B!(v, K, u)
    v .= ν ./ v
    return A_batched_mul_B!(Kv, K, v)
end

function check_convergence(solver::SinkhornSolver)
    return OptimalTransport.check_convergence(
        solver.source,
        solver.cache.u,
        solver.cache.Kv,
        solver.convergence_cache,
        solver.atol,
        solver.rtol,
    )
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
        ε .* reshape(sum(LogExpFunctions.xlogx, γ; dims=(1, 2)), size(γ)[3:end])
    else
        dot_matwise(γ, C)
    end

    return cost
end

"""
    sinkhorn_divergence(μ::AbstractVecOrMat, ν::AbstractVecOrMat, C, ε, alg::Sinkhorn = SinkhornGibbs(); regularization = nothing, plan = nothing, kwargs...)
Compute the Sinkhorn Divergence between finite discrete
measures `μ` and `ν` with respect to a common cost matrix `C`,
entropic regularization parameter `ε` and algorithm `alg`. 
Since there is a single cost matrix, this implementation takes advantage of batch Sinkhorn iterations. 
The default algorithm is the `SinkhornGibbs`.
A pre-computed optimal transport `plan` between `μ` and `ν` may be provided.
The Sinkhorn Divergence is computed as:
```math
\\operatorname{S}_{c,ε}(μ,ν) := \\operatorname{OT}_{c,ε}(μ,ν)
- \\frac{1}{2}(\\operatorname{OT}_{c,ε}(μ,μ) + \\operatorname{OT}_{c,ε}(ν,ν)),
```
where ``\\operatorname{OT}_{c,ε}(μ,ν)``, ``\\operatorname{OT}_{c,ε}(μ,μ)`` and
``\\operatorname{OT}_{c,ε}(ν,ν)`` are the entropically regularized optimal transport cost
between `(μ,ν)`, `(μ,μ)` and `(ν,ν)`, respectively.
The formulation for the Sinkhorn Divergence may have slight variations depending on the paper consulted.
The Sinkhorn Divergence was initially proposed by [^GPC18], although, this package uses the formulation given by
[^FeydyP19], which is also the one used on the Python Optimal Transport package.

[^GPC18]: Aude Genevay, Gabriel Peyré, Marco Cuturi, Learning Generative Models with Sinkhorn Divergences, Proceedings of the Twenty-First International Conference on Artficial Intelligence and Statistics, (AISTATS) 21, 2018
[^FeydyP19]: Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé, and Gabriel Peyré. Interpolating between optimal transport and mmd using sinkhorn divergences. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 2681–2690. PMLR, 2019.
See also: [`sinkhorn2`](@ref)
"""
function sinkhorn_divergence(
    μ::AbstractVecOrMat,
    ν::AbstractVecOrMat,
    C,
    ε,
    alg::Sinkhorn=SinkhornGibbs();
    regularization=nothing,
    plan=nothing,
    kwargs...,
)
    M = max(size(μ, 2), size(ν, 2))
    _μ = size(μ, 2) != M ? repeat(μ, 1, M ÷ size(μ, 2)) : μ
    _ν = size(ν, 2) != M ? repeat(ν, 1, M ÷ size(ν, 2)) : ν
    losses = sinkhorn2(
        hcat(_μ, μ, ν),
        hcat(_ν, μ, ν),
        C,
        ε,
        alg;
        plan=plan,
        regularization=false,
        kwargs...,
    )
    OTμν = losses[1:M]
    OTμ = losses[(M + 1):(M + size(μ, 2))]
    OTν = losses[(M + size(μ, 2) + 1):end]
    return 0 .+ reshape(max.(0, OTμν .- (OTμ .+ OTν) / 2), checksize2(μ, ν))
end

"""
    function sinkhorn_divergence(
        μ,
        ν,
        Cμν,
        Cμ,
        Cν,
        ε,
        alg::Sinkhorn=SinkhornGibbs();
        regularization=nothing,
        plan=nothing,
        kwargs...,
    )
Compute the Sinkhorn Divergence between finite discrete
measures `μ` and `ν` with respect to the precomputed cost matrices `Cμν`,
`Cμμ` and `Cνν`, entropic regularization parameter `ε` and algorithm `alg`.
The default algorithm is the `SinkhornGibbs`.
A pre-computed optimal transport `plan` between `μ` and `ν` may be provided.
See also: [`sinkhorn2`](@ref)
"""
function sinkhorn_divergence(
    μ,
    ν,
    Cμν,
    Cμ,
    Cν,
    ε,
    alg::Sinkhorn=SinkhornGibbs();
    regularization=nothing,
    plan=nothing,
    kwargs...,
)
    if regularization !== nothing
        @warn "`sinkhorn_divergence` does not support the `regularization` keyword argument"
    end

    OTμν = sinkhorn2(μ, ν, Cμν, ε, alg; plan=plan, regularization=false, kwargs...)
    OTμ = sinkhorn2(μ, μ, Cμ, ε, alg; plan=nothing, regularization=false, kwargs...)
    OTν = sinkhorn2(ν, ν, Cν, ε, alg; plan=nothing, regularization=false, kwargs...)
    return max.(0, OTμν - (OTμ + OTν) / 2)
end
