struct SymmetricQuadraticOTNewton{T<:Real,K<:Real,D<:Real} <: QuadraticOT
    θ::T
    κ::K
    δ::D
    armijo_max::Int
end

struct SymmetricQuadraticOTNewtonActiveSet{A<:SymmetricQuadraticOTNewton, ST<:AbstractSparseArray}
    alg::A
    S::ST
end

function SymmetricQuadraticOTNewton(; θ=0.1, κ=0.5, δ=1e-5, armijo_max=50)
    return SymmetricQuadraticOTNewton(θ, κ, δ, armijo_max)
end

function check_convergence(
    μ::AbstractVector,
    cache::QuadraticOTNewtonCache,
    convergence_cache::QuadraticOTConvergenceCache,
    atol::Real,
    rtol::Real,
)
    γ = cache.γ
    norm_diff = norm(vec(sum(γ; dims=2)) .- μ, Inf)
    isconverged =
        norm_diff <
        max(atol, rtol * max(convergence_cache.norm_source, convergence_cache.norm_target))
    return isconverged, norm_diff
end

function descent_dir!(solver::QuadraticOTSolver{<:SymmetricQuadraticOTNewton})
    # unpack solver
    eps = solver.eps
    C = solver.C
    μ = solver.source
    cache = solver.cache
    # unpack cache
    u = cache.u
    δu = cache.δu
    σ = cache.σ
    γ = cache.γ
    G = cache.G
    x = cache.x
    M = cache.M
    N = M
    # Armijo parameters
    δ = solver.alg.δ

    # setup intermediate variables 
    @. γ = u + u' - C
    @. σ = γ ≥ 0
    @. γ = NNlib.relu(γ) / eps

    # setup kernel matrix G
    G = Diagonal(vec(sum(σ; dims=2))) + σ + δ*I

    # cg step
    b = -eps * (vec(sum(γ; dims=2)) .- μ)
    cg!(x, G, b)
    δu .= x
end

function descent_step!(solver::QuadraticOTSolver{<:SymmetricQuadraticOTNewton})
    # unpack solver
    eps = solver.eps
    C = solver.C
    μ = solver.source
    cache = solver.cache
    # unpack cache
    u = cache.u
    δu = cache.δu
    γ = cache.γ

    # Armijo parameters
    θ = solver.alg.θ
    κ = solver.alg.κ
    armijo_max = solver.alg.armijo_max
    armijo_counter = 0

    # dual objective 
    function Φ(u, μ, C, ε)
        return norm(NNlib.relu.(u .+ u' .- C))^2 / 2 - 2*ε * dot(μ, u) 
    end

    # compute directional derivative
    d = -eps * (2*dot(δu, μ)) + eps * dot(γ, δu .+ δu')
    t = 1
    Φ0 = Φ(u, μ, C, eps)
    while (armijo_counter < armijo_max) &&
        (Φ(u + t * δu, μ, C, eps) ≥ Φ0 + t * θ * d)
        t = κ * t
        armijo_counter += 1
    end
    return u .= u + t * δu
end

function solve!(solver::QuadraticOTSolver{<:SymmetricQuadraticOTNewton})
    # unpack solver
    μ = solver.source
    atol = solver.atol
    rtol = solver.rtol
    maxiter = solver.maxiter
    check_convergence = solver.check_convergence
    cache = solver.cache
    convergence_cache = solver.convergence_cache

    isconverged = false
    to_check_step = check_convergence
    for iter in 1:maxiter
        # compute descent direction
        descent_dir!(solver)
        # Newton step
        descent_step!(solver)
        # check source marginal
        # always check convergence after the final iteration
        to_check_step -= 1
        if to_check_step == 0 || iter == maxiter
            # reset counter
            to_check_step = check_convergence

            isconverged, abserror = OptimalTransport.check_convergence(
                μ, μ, cache, convergence_cache, atol, rtol
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

function build_cache(
    ::Type{T},
    ::SymmetricQuadraticOTNewton,
    μ::AbstractVector,
    ν::AbstractVector,
    C::AbstractMatrix,
    ε::Real,
) where {T}
    # create and initialize dual potentials
    u = similar(μ, T, size(μ, 1))
    v = similar(ν, T, size(ν, 1))
    fill!(u, zero(T))
    fill!(v, zero(T))
    δu = similar(u, T)
    δv = similar(v, T)
    # intermediate variables (don't need to be initialised)
    σ = similar(C, T)
    γ = similar(C, T)
    M = size(μ, 1)
    N = size(ν, 1)
    G = similar(u, T, M, M)
    fill!(G, zero(T))
    # initial guess for conjugate gradient 
    x = similar(u, T, M)
    fill!(x, zero(T))
    return QuadraticOTNewtonCache(u, v, δu, δv, σ, γ, G, x, M, N)
end



function build_solver(
    μ::AbstractVector,
    C::AbstractMatrix,
    ε::Real,
    alg::QuadraticOT;
    atol=nothing,
    rtol=nothing,
    check_convergence=1,
    maxiter::Int=100,
)
    # check that source and target marginals have the correct size
    checksize(μ, μ, C)
    # do not use checksize2 since for quadratic OT (at least for now) we do not support batch computations

    # compute type
    T = float(Base.promote_eltype(μ, one(eltype(C)) / ε))

    # build caches
    cache = build_cache(T, alg, μ, μ, C, ε)
    convergence_cache = build_convergence_cache(T, μ, μ)

    # set tolerances
    _atol = atol === nothing ? 0 : atol
    _rtol = rtol === nothing ? (_atol > zero(_atol) ? zero(T) : sqrt(eps(T))) : rtol

    # create solver
    solver = QuadraticOTSolver(
        μ, μ, C, ε, alg, _atol, _rtol, maxiter, check_convergence, cache, convergence_cache
    )
    return solver
end


# interface 

function quadreg(μ, C, ε, alg::SymmetricQuadraticOTNewton; kwargs...)
    solver = build_solver(μ, C, ε, alg; kwargs...)
    solve!(solver)
    γ = plan(solver)
    return γ
end

function plan(solver::QuadraticOTSolver{<:SymmetricQuadraticOTNewton})
    cache = solver.cache
    γ = NNlib.relu.(cache.u .+ cache.u' .- solver.C) / solver.eps
    return γ
end

