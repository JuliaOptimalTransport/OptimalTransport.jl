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
    norm_μ = sum(abs, μ)
    norm_ν = sum(abs, ν)
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
    maxiter::Int=50,
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

Computes the optimal transport plan of histograms `μ` and `ν` with cost matrix `C` and quadratic regularization parameter `ε` using the semismooth Newton algorithm [Lorenz 2019]. 
"""

function quadreg(μ, ν, C, ε, alg::QuadraticOT = QuadraticOTNewton(); kwargs...)
    # build solver
    solver = build_solver(μ, ν, C, ε, alg; kwargs...)

    # perform Sinkhorn algorithm
    solve!(solver)

    # compute optimal transport plan
    γ = plan(solver)

    return γ
end

