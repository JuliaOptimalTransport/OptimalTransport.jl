# Gromov-Wasserstein solver

abstract type EntropicGromovWasserstein end

struct EntropicGromovWassersteinGibbs <: EntropicGromovWasserstein 
    alg_step::Sinkhorn
end

function entropic_gromov_wasserstein(μ::AbstractVector, ν::AbstractVector, Cμ::AbstractMatrix, Cν::AbstractMatrix, ε::Real,
                                    alg::EntropicGromovWassersteinGibbs; atol = nothing, rtol = nothing, check_convergence = 10, maxiter::Int=1_000, kwargs...)
    T = float(Base.promote_eltype(μ, one(eltype(Cμ)) / ε, eltype(Cν)))
    C = similar(Cμ, T, size(μ, 1), size(ν, 1))
    tmp = similar(C)
    plan = similar(C)
    @. plan = μ * ν'

    function get_new_cost!(C, plan, tmp, Cμ, Cν)
        A_batched_mul_B!(tmp, Cμ, plan)
        A_batched_mul_B!(C, tmp, Cν)
    end

    get_new_cost!(C, plan, tmp, Cμ, Cν)
    solver_step = build_solver(μ, ν, C, ε, alg.alg_step; kwargs...)
    to_check_step = check_convergence 
    for iter in 1:maxiter
        # perform Sinkhorn algorithm
        solve!(solver_step)
        # compute optimal transport plan
        plan = sinkhorn_plan(solver)

        to_check_step -= 1
        if to_check_step == 0 || iter == maxiter
            # reset counter
            to_check_step = check_convergence
            
            # TODO: convergence check
            # isconverged, abserror = OptimalTransport.check_convergence(solver)
            # @debug string(solver.alg) *
            #        " (" *
            #        string(iter) *
            #        "/" *
            #        string(maxiter) *
            #        ": absolute error of source marginal = " *
            #        string(maximum(abserror))

            if isconverged
                @debug "$(solver.alg) ($iter/$maxiter): converged"
                break
            end
        end
        update_cost!(solver, C)
    end

    return plan
end

# support for `SinkhornGibbs` and `SinkhornStabilized`
function update_cost!(solver::SinkhornSolver{SinkhornGibbs}, C::AbstractMatrix)
    cache = solver.cache
    @. cache.K = exp(-C / solver.eps)
    newsolver = SinkhornSolver(
        solver.source,
        solver.target,
        C,
        solver.eps,
        solver.alg,
        solver.atol,
        solver.rtol,
        solver.maxiter,
        solver.check_convergence,
        cache,
        solver.convergence_cache,
    )
    return newsolver
end
