# Barycenter solver

struct SinkhornBarycenterSolver{A<:Sinkhorn,M,CT,W,E<:Real,T<:Real,R<:Real,C1,C2}
    source::M
    C::CT
    eps::E
    w::W
    alg::A
    atol::T
    rtol::R
    maxiter::Int
    check_convergence::Int
    cache::C1
    convergence_cache::C2
end

function build_solver(
    μ::AbstractMatrix,
    C::AbstractMatrix,
    ε::Real,
    w::AbstractVector,
    alg::Sinkhorn;
    atol=nothing,
    rtol=nothing,
    check_convergence=10,
    maxiter::Int=1_000,
)
    # check that input marginals are balanced
    checkbalanced(μ)

    size2 = (size(μ, 2),)

    # compute type
    T = float(Base.promote_eltype(μ, one(eltype(C)) / ε))

    # build caches using SinkhornGibbsCache struct (since there is no dependence on ν)
    cache = build_cache(T, alg, size2, μ, C, ε)
    convergence_cache = build_convergence_cache(T, size2, μ)

    # set tolerances
    _atol = atol === nothing ? 0 : atol
    _rtol = rtol === nothing ? (_atol > zero(_atol) ? zero(T) : sqrt(eps(T))) : rtol

    # create solver
    solver = SinkhornBarycenterSolver(
        μ, C, ε, w, alg, _atol, _rtol, maxiter, check_convergence, cache, convergence_cache
    )
    return solver
end

"""
    sinkhorn_barycenter(μ, C, ε, w, alg = SinkhornGibbs(); kwargs...)

Compute the Sinkhorn barycenter for a collection of `N` histograms contained in the columns of `μ`, for a cost matrix `C` of size `(size(μ, 1), size(μ, 1))`, relative weights `w` of size `N`, and entropic regularisation parameter `ε`.
Returns the entropically regularised barycenter of the `μ`, i.e. the histogram `ρ` of length `size(μ, 1)` that solves

```math
\\min_{\\rho \\in \\Sigma} \\sum_{i = 1}^N w_i \\operatorname{OT}_{\\varepsilon}(\\mu_i, \\rho)
```

where ``\\operatorname{OT}_{ε}(\\mu, \\nu) = \\inf_{\\gamma \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle + \\varepsilon \\Omega(\\gamma)``
is the entropic optimal transport loss with cost ``C`` and regularisation ``\\epsilon``.
"""
function sinkhorn_barycenter(μ, C, ε, w, alg::Sinkhorn; kwargs...)
    solver = build_solver(μ, C, ε, w, alg; kwargs...)
    solve!(solver)
    return solution(solver)
end
