struct SinkhornStabilizedCache{U,V,KT,M}
    u::U
    v::V
    alpha::U
    beta::V
    Ktilde::KT
    Ktildebeta::U
    tmp::U
    norm_mu::M
end

function SinkhornStabilizedCache(μ, ν, C, ε)
    # check that source and target marginals have the same mass
    checkbalanced(μ, ν)

    # compute the element type of the caches
    T = float(Base.promote_eltype(μ, ν, C, inv(ε)))

    # compute the Gibbs kernel (has to be mutable since it will be modified)
    K̃ = similar(C, T)
    @. K̃ = exp(-C / ε)

    # dual potentials and caches
    u = similar(μ, T)
    v = similar(ν, T)
    α = similar(μ, T)
    β = similar(ν, T)
    K̃β = similar(μ, T)
    tmp = similar(μ, T)

    # evaluate L1 norm of `μ` for convergence checks
    normμ = sum(abs, μ)

    # initializations
    fill!(u, zero(T))
    fill!(v, zero(T))
    fill!(α, one(T))
    fill!(β, one(T))

    # create cache
    cache = SinkhornStabilizedCache(u, v, α, β, K̃, K̃β, tmp, normμ)

    return cache
end

"""
    sinkhorn_stabilized_epsscaling(
        μ, ν, C, ε; scaling_factor=0.5, scaling_steps=5, kwargs...
    )

Compute the optimal transport plan for the entropically regularized optimal transport
problem with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, and entropic regularisation parameter `ε`.

The function uses the log-domain stabilized Sinkhorn algorithm with `scaling_steps`
ε-scaling steps with scaling factor `scaling_factor`. Using [`sinkhorn_stabilized`](@ref),
it sequentially solves the entropically regularized optimal transport with regularization
parameters
```math
\\varepsilon_i := \\varepsilon \\lambda^{i-k}, \\qquad (i = 1,\\ldots,k),
```
where ``\\lambda`` is the scaling factor and ``k`` the number of scaling steps.

The other keyword arguments supported here are the same as those in the
[`sinkhorn_stabilized`](@ref) function.

See also: [`sinkhorn_stabilized`](@ref), [`sinkhorn`](@ref)

# References

Schmitzer, B. (2019).
[Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems](https://doi.org/10.1137/16m1106018).
SIAM Journal on Scientific Computing, 41(3), A1443–A1481.
"""
function sinkhorn_stabilized_epsscaling(
    μ, ν, C, ε; lambda=nothing, scaling_factor=lambda, k=nothing, scaling_steps=k, kwargs...
)
    if lambda !== nothing
        Base.depwarn(
            "keyword argument `lambda` is deprecated, please use `scaling_factor`",
            :sinkhorn_stabilized_epsscaling,
        )
    end
    if k !== nothing
        Base.depwarn(
            "keyword argument `k` is deprecated, please use `scaling_steps`",
            :sinkhorn_stabilized_epsscaling,
        )
    end

    _scaling_factor = scaling_factor === nothing ? 1//2 : scaling_factor
    _scaling_steps = scaling_steps === nothing ? 5 : scaling_steps

    # initial regularization parameter
    εstep = ε * _scaling_factor^(1 - _scaling_steps)
    @debug "ε-scaling Sinkhorn algorithm: ε = $εstep"

    # initialize cache and perform stabilized Sinkhorn algorithm
    cache = SinkhornStabilizedCache(μ, ν, C, εstep)
    solve!(cache, μ, ν, C, εstep; kwargs...)

    for _ in 2:(_scaling_steps - 1)
        εstep *= _scaling_factor
        @debug "ε-scaling Sinkhorn algorithm: ε = $εstep"

        # re-run Sinkhorn algorithm with smaller regularization parameter
        updateKtilde!(cache, C, εstep)
        solve!(cache, μ, ν, C, εstep; kwargs...)
    end

    # re-run Sinkhorn algorithm with desired regularization parameter
    @debug "ε-scaling Sinkhorn algorithm: ε = $ε"
    updateKtilde!(cache, C, ε)
    solve!(cache, μ, ν, C, ε; kwargs...)

    # compute final plan
    updateKtilde!(cache, C, ε)
    γ = cache.Ktilde

    return γ
end

"""
    sinkhorn_stabilized(
        μ, ν, C, ε;
        absorb_tol=1_000,
        maxiter=1_000,
        atol=0,
        rtol=atol > 0 ? 0 : √eps,
        check_convergence=10,
        maxiter=1_000,
    )

Compute the optimal transport plan for the entropically regularized optimal transport
problem with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, and entropic regularisation parameter `ε`.

The function uses the log-domain stabilized Sinkhorn algorithm with an absorption tolerance
`absorb_tol`.

Every `check_convergence` steps it is assessed if the algorithm is converged by checking if
the iterate of the transport plan `G` satisfies
```julia
isapprox(sum(G; dims=2), μ; atol=atol, rtol=rtol, norm=x -> norm(x, 1))
```
The default `rtol` depends on the types of `μ`, `ν`, `C`, and `ε`. After `maxiter`
iterations, the computation is stopped.

See also: [`sinkhorn`](@ref)

# References

Schmitzer, B. (2019).
[Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems](https://doi.org/10.1137/16m1106018).
SIAM Journal on Scientific Computing, 41(3), A1443–A1481.
"""
function sinkhorn_stabilized(
    μ, ν, C, ε; tol=nothing, atol=tol, return_duals=nothing, kwargs...
)
    # deprecation warnings
    if tol !== nothing
        Base.depwarn(
            "keyword argument `tol` is deprecated, please use `atol` and `rtol`",
            :sinkhorn_stabilized,
        )
    end
    if return_duals !== nothing
        Base.depwarn(
            "keyword argument `return_duals` is deprecated and will be removed",
            :sinkhorn_stabilized,
        )
    end

    # create cache
    cache = SinkhornStabilizedCache(μ, ν, C, ε)
    solve!(cache, μ, ν, C, ε; atol=atol, kwargs...)

    # return "dual potentials" (in log space and scaled by 1/ε) if requested
    if return_duals !== nothing && return_duals
        return cache.u, cache.v
    end

    # compute optimal transport plan
    updateKtilde!(cache, C, ε)
    γ = cache.Ktilde

    return γ
end

# perform stabilized Sinkhorn algorithm
function solve!(
    cache::SinkhornStabilizedCache,
    μ,
    ν,
    C,
    ε;
    atol=nothing,
    rtol=nothing,
    absorb_tol::Real=1_000,
    maxiter::Int=1_000,
    check_convergence::Int=10,
)
    # unpack
    u = cache.u
    v = cache.v
    α = cache.alpha
    β = cache.beta
    K̃ = cache.Ktilde
    K̃β = cache.Ktildebeta
    tmp = cache.tmp
    norm_μ = cache.norm_mu

    # compute default tolerances
    T = eltype(u)
    _atol = atol === nothing ? 0 : atol
    _rtol = rtol === nothing ? (_atol > zero(_atol) ? zero(T) : sqrt(eps(T))) : rtol

    # initial iterate
    mul!(K̃β, K̃, β)

    isconverged = false
    to_check_step = check_convergence
    for iter in 1:maxiter
        # reduce counter
        to_check_step -= 1

        # absorption step
        if maximum(abs, u) > absorb_tol || maximum(abs, v) > absorb_tol
            @debug "stabilized Sinkhorn algorithm (" *
                   string(iter) *
                   "/" *
                   string(maxiter) *
                   ": absorbing `α` and `β` into `u` and `v`"

            # absorb `α` and `β` in `u` and `v`
            absorb!(cache, ε)

            # update iterates
            updateKtilde!(cache, C, ε)
            fill!(β, one(eltype(β)))
            mul!(K̃β, K̃, β)
        end

        # compute next iterate
        α .= μ ./ K̃β
        mul!(β, K̃', α)
        β .= ν ./ β
        mul!(K̃β, K̃, β)

        # check source marginal
        # always check convergence after the final iteration
        if to_check_step == 0 || iter == maxiter
            # reset counter
            to_check_step = check_convergence

            # do not overwrite `Kβ` but reuse it for computing `α` if not converged
            tmp .= α .* K̃β
            norm_αK̃β = sum(abs, tmp)
            tmp .= abs.(μ .- tmp)
            norm_diff = sum(tmp)

            @debug "stabilized Sinkhorn algorithm (" *
                   string(iter) *
                   "/" *
                   string(maxiter) *
                   ": absolute error of source marginal = " *
                   string(norm_diff)

            isconverged = norm_diff < max(_atol, _rtol * max(norm_μ, norm_αK̃β))
            if isconverged
                @debug "Stabilized Sinkhorn algorithm ($iter/$maxiter): converged"
                break
            end
        end
    end

    # absorb `α` and `β` into `u` and `v`
    absorb!(cache, ε)

    if !isconverged
        @warn "Stabilized Sinkhorn algorithm ($maxiter/$maxiter): not converged"
    end

    return nothing
end

# absorption step
function absorb!(cache::SinkhornStabilizedCache, ε)
    # unpack
    u = cache.u
    v = cache.v
    α = cache.alpha
    β = cache.beta

    # update dual potentials: absorb `α` and `β` in `u` and `v`
    u .+= ε .* log.(α)
    v .+= ε .* log.(β)

    return nothing
end

# update kernel
function updateKtilde!(cache::SinkhornStabilizedCache, C, ε)
    @. cache.Ktilde = exp(-(C - cache.u - cache.v') / ε)
    return nothing
end
