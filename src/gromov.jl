# Gromov-Wasserstein solver

abstract type EntropicGromovWasserstein end

struct EntropicGromovWassersteinSinkhorn <: EntropicGromovWasserstein
    alg_step::Sinkhorn
end

"""
    entropic_gromov_wasserstein(
        μ, ν, Cμ, Cν, ε, alg=EntropicGromovWassersteinSinkhorn(SinkhornGibbs()); 
        atol = nothing, rtol = nothing, check_convergence = 10, maxiter = 1_000, kwargs...
    )

Computes the transport map for the entropically regularized Gromov-Wasserstein optimal transport problem with source and target 
marginals `μ` and `ν` and corresponding cost matrices `Cμ` and `Cν`. That is, we seek `γ` a local minimizer of 
```math
    \\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\sum_{i, j, i', j'} |C^{(\\mu)}_{i,i'} - C^{(\\nu)}_{j,j'}| \\gamma_{i,j} \\gamma_{i',j'} + \\varepsilon \\Omega(\\gamma),
```
where ``\\Omega(\\gamma)`` is the entropic regularization term, see e.g. [`sinkhorn`](@ref). 

This function employs the iterative method described in (Section 10.6.4, [^PC19]), which solves a series of Sinkhorn iteration sub-problems to arrive at a solution. Note that the Gromov-Wasserstein problem is non-convex owing to the cross-terms in the 
objective function, and thus in general one is guaranteed to arrive at a local optimum. 

Every `check_convergence` steps, the current iteration of `γ` is compared with `γ_prev` (the previous iteration from `check_convergence` ago). 
The quantity ``\\| \\gamma - \\gamma_\\text{prev} \\|_1`` is compared against `atol` and `rtol`. 

[^PC19]: Peyré, G. and Cuturi, M., 2019. Computational optimal transport: With applications to data science. Foundations and Trends® in Machine Learning, 11(5-6), pp.355-607. 

See also: [`sinkhorn`](@ref)
"""
function entropic_gromov_wasserstein(
    μ::AbstractVector,
    ν::AbstractVector,
    Cμ::AbstractMatrix,
    Cν::AbstractMatrix,
    ε::Real,
    alg::EntropicGromovWasserstein=EntropicGromovWassersteinSinkhorn(SinkhornGibbs());
    atol=nothing,
    rtol=nothing,
    check_convergence=10,
    maxiter::Int=1_000,
    kwargs...,
)
    T = float(Base.promote_eltype(μ, one(eltype(Cμ)) / ε, eltype(Cν)))
    C = similar(Cμ, T, size(μ, 1), size(ν, 1))
    tmp = similar(C)
    plan = similar(C)
    @. plan = μ * ν'
    plan_prev = similar(C)
    plan_prev .= plan
    norm_plan = sum(plan)

    _atol = atol === nothing ? 0 : atol
    _rtol = rtol === nothing ? (_atol > zero(_atol) ? zero(T) : sqrt(eps(T))) : rtol

    function get_new_cost!(C, plan, tmp, Cμ, Cν)
        A_batched_mul_B!(tmp, Cμ, plan)
        return A_batched_mul_B!(C, tmp, -4Cν)
        # seems to be a missing factor of 4 (or something like that...) compared to the POT implementation?
        # added the factor of 4 here to ensure reproducibility for the same value of ε.
        # https://github.com/PythonOT/POT/blob/9412f0ad1c0003e659b7d779bf8b6728e0e5e60f/ot/gromov.py#L247
    end

    get_new_cost!(C, plan, tmp, Cμ, Cν)
    to_check_step = check_convergence

    isconverged = false
    for iter in 1:maxiter
        # perform Sinkhorn algorithm
        solver = build_solver(μ, ν, C, ε, alg.alg_step; kwargs...)
        solve!(solver)
        # compute optimal transport plan
        plan = sinkhorn_plan(solver)

        to_check_step -= 1
        if to_check_step == 0 || iter == maxiter
            # reset counter
            to_check_step = check_convergence
            isconverged = sum(abs, plan - plan_prev) < max(_atol, _rtol * norm_plan)
            if isconverged
                @debug "Gromov Wasserstein with $(solver.alg) ($iter/$maxiter): converged"
                break
            end
            plan_prev .= plan
        end
        get_new_cost!(C, plan, tmp, Cμ, Cν)
    end

    return plan
end
