
"""
    Semi-smooth Newton method (Algorithm 2 of Lorenz et al. 2019) for solving quadratically regularised optimal transport

    See also: [`QuadraticOTNewton`](@ref), [`quadreg`](@ref)

    # References
    
    Lorenz, Dirk A., Paul Manns, and Christian Meyer. [Quadratically regularized optimal transport.](https://doi.org/10.1007/s00245-019-09614-w) Applied Mathematics & Optimization 83.3 (2021): 1919-1949.
"""
struct QuadraticOTNewton{T<:Real,K<:Real,D<:Real} <: QuadraticOT
    θ::T
    κ::K
    δ::D
end

"""
    QuadraticOTNewton(θ = 0.1, κ = 0.5, δ = 1e-5)

    Semi-smooth Newton method (Algorithm 2 of Lorenz et al. 2019) with Armijo parameters `θ` and `κ`, and conjugate gradient regularisation `δ`. 
    
    See also: [`QuadraticOTNewton`](@ref), [`quadreg`](@ref)
"""
function QuadraticOTNewton(θ=0.1, κ=0.5, δ=1e-5)
    return QuadraticOTNewton(θ, κ, δ)
end

Base.show(io::IO, ::QuadraticOTNewton) = print(io, "Semi-smooth Newton algorithm")

struct QuadraticOTNewtonCache{U,V,C,P,GT,X}
    u::U
    v::V
    δu::U
    δv::V
    σ::C
    γ::P
    G::GT
    x::X
    M::Int
    N::Int
end

function build_cache(
    ::Type{T},
    ::QuadraticOTNewton,
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
    δu = zeros(T, size(u))
    δv = zeros(T, size(v))
    # intermediate variables (don't need to be initialised)
    σ = similar(C, T)
    γ = similar(C, T)
    M = size(μ, 1)
    N = size(ν, 1)
    G = zeros(T, M + N, M + N)
    # initial guess for conjugate gradient 
    x = zeros(T, M + N)
    return QuadraticOTNewtonCache(u, v, δu, δv, σ, γ, G, x, M, N)
end

function check_convergence(
    μ::AbstractVector,
    ν::AbstractVector,
    cache::QuadraticOTNewtonCache,
    convergence_cache::QuadraticOTConvergenceCache,
    atol::Real,
    rtol::Real,
)
    γ = cache.γ
    norm_diff = max(
        norm(vec(sum(γ; dims=2)) .- μ, Inf), norm(vec(sum(γ; dims=1) .- ν), Inf)
    )
    isconverged =
        norm_diff <
        max(atol, rtol * max(convergence_cache.norm_source, convergence_cache.norm_target))
    return isconverged, norm_diff
end

function descent_dir!(solver::QuadraticOTSolver{<:QuadraticOTNewton})
    # unpack solver
    eps = solver.eps
    C = solver.C
    μ = solver.source
    ν = solver.target
    cache = solver.cache
    # unpack cache
    u = cache.u
    v = cache.v
    δu = cache.δu
    δv = cache.δv
    σ = cache.σ
    γ = cache.γ
    G = cache.G
    x = cache.x
    M = cache.M
    N = cache.N
    # Armijo parameters
    δ = solver.alg.δ

    # setup intermediate variables 
    @. γ = u + v' - C
    @. σ = ifelse(γ ≥ 0, 1, 0)
    @. γ = NNlib.relu(γ) / eps

    # setup kernel matrix G
    G[diagind(G)[1:M]] .= vec(sum(σ; dims=2))
    G[diagind(G)[(M + 1):end]] .= vec(sum(σ; dims=1))
    G[1:M, (M + 1):end] .= σ
    G[(M + 1):end, 1:M] .= σ'
    G[diagind(G)] .+= δ # regularise cg

    # cg step
    b = -eps * vcat(vec(sum(γ; dims=2)) .- μ, vec(sum(γ; dims=1)) .- ν)
    cg!(x, G, b)
    δu .= x[1:M]
    return δv .= x[(M + 1):end]
end

function descent_step!(solver::QuadraticOTSolver{<:QuadraticOTNewton})
    # unpack solver
    eps = solver.eps
    C = solver.C
    μ = solver.source
    ν = solver.target
    cache = solver.cache
    # unpack cache
    u = cache.u
    v = cache.v
    δu = cache.δu
    δv = cache.δv
    γ = cache.γ

    # Armijo parameters
    θ = solver.alg.θ
    κ = solver.alg.κ

    # dual objective 
    function Φ(u, v, μ, ν, C, ε)
        return norm(NNlib.relu.(u .+ v' .- C))^2 / 2 - ε * dot(μ, u) - ε * dot(ν, v)
    end

    # compute directional derivative
    d = -eps * (dot(δu, μ) + dot(δv, ν)) + eps * dot(γ, δu .+ δv')
    t = 1
    Φ0 = Φ(u, v, μ, ν, C, eps)
    while Φ(u + t * δu, v + t * δv, μ, ν, C, eps) ≥ Φ0 + t * θ * d
        t = κ * t
    end
    u .= u + t * δu
    return v .= v + t * δv
end

function solve!(solver::QuadraticOTSolver{<:QuadraticOTNewton})
    # unpack solver
    μ = solver.source
    ν = solver.target
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
                μ, ν, cache, convergence_cache, atol, rtol
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

# interface 

function plan(solver::QuadraticOTSolver{<:QuadraticOTNewton})
    cache = solver.cache
    γ = NNlib.relu.(cache.u .+ cache.v' .- solver.C) / solver.eps
    return γ
end
