struct SymmetricSinkhornSolver{A<:Sinkhorn,M,CT,E<:Real,T<:Real,R<:Real,C1,C2}
    source::M
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

struct SymmetricSinkhornGibbs <: Sinkhorn end

Base.show(io::IO, ::SymmetricSinkhornGibbs) = print(io, "Symmetric Sinkhorn algorithm")

struct SymmetricSinkhornGibbsCache{U,KT}
    u::U
    K::KT
    Kv::U
end

function build_cache(
    ::Type{T},
    ::SymmetricSinkhornGibbs,
    size2::Tuple,
    μ::AbstractVecOrMat,
    C::AbstractMatrix,
    ε::Real,
) where {T}
    K = similar(C, T)
    @. K = exp(-C / ε)

    u = similar(μ, T, size(μ, 1), size2...)
    fill!(u, one(T))

    Kv = similar(u)

    return SymmetricSinkhornGibbsCache(u, K, Kv)
end

function check_convergence(solver::SymmetricSinkhornSolver)
    A_batched_mul_B!(solver.cache.Kv, solver.cache.K, solver.cache.u)
    return OptimalTransport.check_convergence(
        solver.source,
        solver.cache.u,
        solver.cache.Kv,
        solver.convergence_cache,
        solver.atol,
        solver.rtol,
    )
end

function build_solver(
    μ::AbstractVecOrMat,
    C::AbstractMatrix,
    ε::Real,
    alg::SymmetricSinkhornGibbs;
    atol=nothing,
    rtol=nothing,
    check_convergence=1,
    maxiter::Int=25,
)
    size2 = size(μ)[2:end]

    # compute type
    T = float(Base.promote_eltype(μ, one(eltype(C)) / ε))

    # build caches using SinkhornGibbsCache struct (since there is no dependence on ν)
    cache = build_cache(T, alg, size2, μ, C, ε)
    convergence_cache = build_convergence_cache(T, size2, μ)

    # set tolerances
    _atol = atol === nothing ? 0 : atol
    _rtol = rtol === nothing ? (_atol > zero(_atol) ? zero(T) : sqrt(eps(T))) : rtol

    # create solver
    solver = SymmetricSinkhornSolver(
        μ, C, ε, alg, _atol, _rtol, maxiter, check_convergence, cache, convergence_cache
    )
    return solver
end

function init_step!(solver::SymmetricSinkhornSolver{SymmetricSinkhornGibbs})
    source = solver.source
    cache = solver.cache
    u = cache.u
    K = cache.K
    Kv = cache.Kv
    return A_batched_mul_B!(Kv, K, u)
end

prestep!(::SymmetricSinkhornSolver{SymmetricSinkhornGibbs}, ::Int) = nothing

function step!(solver::SymmetricSinkhornSolver{SymmetricSinkhornGibbs}, iter::Int)
    source = solver.source
    cache = solver.cache
    u = cache.u
    K = cache.K
    Kv = cache.Kv
    @. u = sqrt(source * u / Kv)
    return A_batched_mul_B!(Kv, K, u)
end

function sinkhorn_plan(solver::SymmetricSinkhornSolver{SymmetricSinkhornGibbs})
    cache = solver.cache
    return sinkhorn_plan(cache.u, cache.u, cache.K)
end

function sinkhorn(μ, C, ε, alg::SymmetricSinkhornGibbs; kwargs...)
    # build solver
    solver = build_solver(μ, C, ε, alg; kwargs...)

    # perform Sinkhorn algorithm
    solve!(solver)

    # compute optimal transport plan
    γ = sinkhorn_plan(solver)

    return γ
end

function sinkhorn2(
    μ, C, ε, alg::SymmetricSinkhornGibbs; regularization=false, plan=nothing, kwargs...
)
    cost = if regularization && plan === nothing
        # special case where we can take advantage of dual objective formula 
        # build solver
        solver = build_solver(μ, C, ε, alg; kwargs...)
        # perform Sinkhorn algorithm
        solve!(solver)
        # return loss
        cache = solver.cache
        sinkhorn_dual_objective(cache.u, cache.u, cache.Kv, cache.K, solver.eps)
    else
        γ = if plan === nothing
            sinkhorn(μ, C, ε, alg; kwargs...)
        else
            size(plan) == size(C) || error(
                "optimal transport plan `plan` and cost matrix `C` must be of the same size",
            )
            plan
        end
        cost = sinkhorn_cost_from_plan(γ, C, ε; regularization=regularization)
        cost
    end
    return cost
end
