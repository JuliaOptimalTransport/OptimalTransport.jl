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

function solve!(solver::Union{SinkhornSolver, SinkhornBarycenterSolver})
    # unpack solver
    atol = solver.atol
    rtol = solver.rtol
    maxiter = solver.maxiter
    check_convergence = solver.check_convergence
    cache = solver.cache
    convergence_cache = solver.convergence_cache

    isconverged = false
    to_check_step = check_convergence
    init_step!(solver)
    for iter in 1:maxiter
        # computations before the Sinkhorn iteration (e.g., absorption step)
        prestep!(solver, iter)
        # perform Sinkhorn iteration
        step!(solver, iter)

        # check source marginal
        # always check convergence after the final iteration
        to_check_step -= 1
        if to_check_step == 0 || iter == maxiter
            # reset counter
            to_check_step = check_convergence

            isconverged, abserror = OptimalTransport.check_convergence(solver)
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
