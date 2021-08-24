# algorithm

abstract type QuadraticOT end

# solver 

struct QuadraticOTSolver{A<:QuadraticOT,M,N,CT,E<:Real,T<:Real,R<:Real,C1,C2}
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

struct QuadraticOTConvergenceCache{N<:Real}
    norm_source::N
    norm_target::N
end

function build_convergence_cache(::Type{T}, μ::AbstractVector, ν::AbstractVector) where {T}
    norm_μ = norm(μ, Inf)
    norm_ν = norm(ν, Inf)
    return QuadraticOTConvergenceCache(norm_μ, norm_ν)
end

function build_solver(
    μ::AbstractVector,
    ν::AbstractVector,
    C::AbstractMatrix,
    ε::Real,
    alg::QuadraticOT;
    atol=nothing,
    rtol=nothing,
    check_convergence=1,
    maxiter::Int=100,
)
    # check that source and target marginals have the correct size and are balanced
    checksize(μ, ν, C)
    checkbalanced(μ, ν)
    # do not use checksize2 since for quadratic OT (at least for now) we do not support batch computations

    # compute type
    T = float(Base.promote_eltype(μ, ν, one(eltype(C)) / ε))

    # build caches
    cache = build_cache(T, alg, μ, ν, C, ε)
    convergence_cache = build_convergence_cache(T, μ, ν)

    # set tolerances
    _atol = atol === nothing ? 0 : atol
    _rtol = rtol === nothing ? (_atol > zero(_atol) ? zero(T) : sqrt(eps(T))) : rtol

    # create solver
    solver = QuadraticOTSolver(
        μ, ν, C, ε, alg, _atol, _rtol, maxiter, check_convergence, cache, convergence_cache
    )

    return solver
end

"""
    quadreg(μ, ν, C, ε, alg::QuadraticOT; kwargs...)

Computes the optimal transport plan of histograms `μ` and `ν` with cost matrix `C` and quadratic regularization parameter `ε`. 

The optimal transport plan `γ` is of the same size as `C` and solves 

```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma),
```
where ``\\Omega(\\gamma) = \\frac{1}{2} \\sum_{i,j} \\gamma_{i,j}^2`` is the quadratic 
regularization term.

Every `check_convergence` steps it is assessed if the algorithm is converged by checking if
the iterate of the transport plan `γ` satisfies
```julia
    norm_diff < max(atol, rtol * max(norm(μ, Inf), norm(ν, Inf)))
```
where
```math
    \text{normdiff} = \\max\\{ \\| \\gamma \\mathbf{1} - \\mu \\|_\\infty , \\|  \\gamma^\\top \\mathbf{1} - \\nu \\|_\\infty  \\} . 
```
After `maxiter` iterations, the computation is stopped.

Note that unlike in the case of Sinkhorn's algorithm for the entropic regularisation, batch computation of optimal transport is not supported for the quadratic regularisation. 

See also: [`sinkhorn`](@ref), [`QuadraticOTNewton`](@ref)
"""
function quadreg(μ, ν, C, ε, alg::QuadraticOT=QuadraticOTNewton(); kwargs...)
    # build solver
    solver = build_solver(μ, ν, C, ε, alg; kwargs...)

    # perform Sinkhorn algorithm
    solve!(solver)

    # compute optimal transport plan
    γ = plan(solver)

    return γ
end
