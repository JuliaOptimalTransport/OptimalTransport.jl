"""
    sinkhorn_gibbs(
        μ, ν, K; atol=0, rtol=atol > 0 ? 0 : √eps, check_convergence=10, maxiter=1_000
    )

Compute the dual potentials for the entropically regularized optimal transport problem
with source and target marginals `μ` and `ν` and Gibbs kernel `K` using the Sinkhorn
algorithm.

The Gibbs kernel `K` is defined as
```math
K = \\exp(-C / \\varepsilon),
```
where ``C`` is the cost matrix and ``\\varepsilon`` the entropic regularization parameter.
The corresponding optimal transport plan can be computed from the dual potentials ``u``
and ``v`` as
```math
\\gamma = \\operatorname{diag}(u) K \\operatorname{diag}(v).
```
Every `check_convergence` steps it is assessed if the algorithm is converged by checking if
the iterate of the transport plan `G` satisfies
```julia
isapprox(sum(G; dims=2), μ; atol=atol, rtol=rtol, norm=x -> norm(x, 1))
```
The default `rtol` depends on the types of `μ`, `ν`, and `K`. After `maxiter` iterations,
the computation is stopped.

Batch computations for multiple histograms with a common Gibbs kernel `K` can be performed
by passing `μ` or `ν` as matrices whose columns correspond to histograms. It is required
that the number of source and target marginals is equal or that a single source or single
target marginal is provided (either as matrix or as vector). The optimal transport plans are
returned as three-dimensional array where `γ[:, :, i]` is the optimal transport plan for the
`i`th pair of source and target marginals.
"""
function sinkhorn_gibbs(
    μ,
    ν,
    K;
    tol=nothing,
    atol=tol,
    rtol=nothing,
    check_marginal_step=nothing,
    check_convergence=check_marginal_step,
    maxiter::Int=1_000,
)
    if tol !== nothing
        Base.depwarn(
            "keyword argument `tol` is deprecated, please use `atol` and `rtol`",
            :sinkhorn_gibbs,
        )
    end
    if check_marginal_step !== nothing
        Base.depwarn(
            "keyword argument `check_marginal_step` is deprecated, please use `check_convergence`",
            :sinkhorn_gibbs,
        )
    end

    # checks
    size2 = checksize2(μ, ν)
    checkbalanced(μ, ν)

    # set default values of tolerances
    T = float(Base.promote_eltype(μ, ν, K))
    _atol = atol === nothing ? 0 : atol
    _rtol = rtol === nothing ? (_atol > zero(_atol) ? zero(T) : sqrt(eps(T))) : rtol

    # initialize iterates
    u = similar(μ, T, size(μ, 1), size2...)
    v = similar(ν, T, size(ν, 1), size2...)
    fill!(v, one(T))

    # arrays for convergence check
    Kv = similar(u)
    mul!(Kv, K, v)
    tmp = similar(u)
    norm_μ = μ isa AbstractVector ? sum(abs, μ) : sum(abs, μ; dims=1)
    if u isa AbstractMatrix
        tmp2 = similar(u)
        norm_uKv = similar(u, 1, size2...)
        norm_diff = similar(u, 1, size2...)
        _isconverged = similar(u, Bool, 1, size2...)
    end

    isconverged = false
    check_step = check_convergence === nothing ? 10 : check_convergence
    to_check_step = check_step
    for iter in 1:maxiter
        # reduce counter
        to_check_step -= 1

        # compute next iterate
        u .= μ ./ Kv
        mul!(v, K', u)
        v .= ν ./ v
        mul!(Kv, K, v)

        # check source marginal
        # always check convergence after the final iteration
        if to_check_step <= 0 || iter == maxiter
            # reset counter
            to_check_step = check_step

            # do not overwrite `Kv` but reuse it for computing `u` if not converged
            tmp .= u .* Kv
            if u isa AbstractMatrix
                tmp2 .= abs.(tmp)
                sum!(norm_uKv, tmp2)
            else
                norm_uKv = sum(abs, tmp)
            end
            tmp .= abs.(μ .- tmp)
            if u isa AbstractMatrix
                sum!(norm_diff, tmp)
            else
                norm_diff = sum(tmp)
            end

            @debug "Sinkhorn algorithm (" *
                   string(iter) *
                   "/" *
                   string(maxiter) *
                   ": absolute error of source marginal = " *
                   string(maximum(norm_diff))

            # check stopping criterion
            isconverged = if u isa AbstractMatrix
                @. _isconverged = norm_diff < max(_atol, _rtol * max(norm_μ, norm_uKv))
                all(_isconverged)
            else
                norm_diff < max(_atol, _rtol * max(norm_μ, norm_uKv))
            end
            if isconverged
                @debug "Sinkhorn algorithm ($iter/$maxiter): converged"
                break
            end
        end
    end

    if !isconverged
        @warn "Sinkhorn algorithm ($maxiter/$maxiter): not converged"
    end

    return u, v
end

"""
    sinkhorn(
        μ, ν, C, ε; atol=0, rtol=atol > 0 ? 0 : √eps, check_convergence=10, maxiter=1_000
    )

Compute the optimal transport plan for the entropically regularized optimal transport
problem with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, and entropic regularization parameter `ε`.

The optimal transport plan `γ` is of the same size as `C` and solves
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma),
```
where ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{i,j} \\log \\gamma_{i,j}`` is the entropic
regularization term.

Every `check_convergence` steps it is assessed if the algorithm is converged by checking if
the iterate of the transport plan `G` satisfies
```julia
isapprox(sum(G; dims=2), μ; atol=atol, rtol=rtol, norm=x -> norm(x, 1))
```
The default `rtol` depends on the types of `μ`, `ν`, and `C`. After `maxiter` iterations,
the computation is stopped.

Batch computations for multiple histograms with a common cost matrix `C` can be performed by
passing `μ` or `ν` as matrices whose columns correspond to histograms. It is required that
the number of source and target marginals is equal or that a single source or single target
marginal is provided (either as matrix or as vector). The optimal transport plans are
returned as three-dimensional array where `γ[:, :, i]` is the optimal transport plan for the
`i`th pair of source and target marginals.

See also: [`sinkhorn2`](@ref)
"""
function sinkhorn(μ, ν, C, ε; kwargs...)
    # compute Gibbs kernel
    K = @. exp(-C / ε)

    # compute dual potentials
    u, v = sinkhorn_gibbs(μ, ν, K; kwargs...)
    return K .* add_singleton(u, Val(2)) .* add_singleton(v, Val(1))
end

"""
    sinkhorn2(μ, ν, C, ε; regularization=false, plan=nothing, kwargs...)

Solve the entropically regularized optimal transport problem with source and target
marginals `μ` and `ν`, cost matrix `C` of size `(length(μ), length(ν))`, and entropic
regularization parameter `ε`, and return the optimal cost.

A pre-computed optimal transport `plan` may be provided. The other keyword arguments
supported here are the same as those in the [`sinkhorn`](@ref) function.

!!! note
    As the `sinkhorn2` function in the Python Optimal Transport package, this function
    returns the optimal transport cost without the regularization term. The cost
    with the regularization term can be computed by setting `regularization=true`.

See also: [`sinkhorn`](@ref)
"""
function sinkhorn2(μ, ν, C, ε; regularization=false, plan=nothing, kwargs...)
    γ = if plan === nothing
        sinkhorn(μ, ν, C, ε; kwargs...)
    else
        # check dimensions
        size(C) == (size(μ, 1), size(ν, 1)) || error(
            "cost matrix `C` must be of size `(size(μ, dims = 1), size(ν, dims = 1))`",
        )
        (size(plan, 1), size(plan, 2)) == size(C) || error(
            "optimal transport plan `plan` and cost matrix `C` must be of the same size",
        )
        plan
    end
    cost = if regularization
        dot_matwise(γ, C) .+
        ε * reshape(sum(LogExpFunctions.xlogx, γ; dims=(1, 2)), size(γ)[3:end])
    else
        dot_matwise(γ, C)
    end

    return cost
end

"""
    sinkhorn_unbalanced(μ, ν, C, λ1::Real, λ2::Real, ε; kwargs...)

Compute the optimal transport plan for the unbalanced entropically regularized optimal
transport problem with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, entropic regularization parameter `ε`, and marginal relaxation
terms `λ1` and `λ2`.

The optimal transport plan `γ` is of the same size as `C` and solves
```math
\\inf_{\\gamma} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma)
+ \\lambda_1 \\operatorname{KL}(\\gamma 1 | \\mu)
+ \\lambda_2 \\operatorname{KL}(\\gamma^{\\mathsf{T}} 1 | \\nu),
```
where ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{i,j} \\log \\gamma_{i,j}`` is the entropic
regularization term and ``\\operatorname{KL}`` is the Kullback-Leibler divergence.

The keyword arguments supported here are the same as those in the `sinkhorn_unbalanced`
for unbalanced optimal transport problems with general soft marginal constraints.
"""
function sinkhorn_unbalanced(
    μ, ν, C, λ1::Real, λ2::Real, ε; proxdiv_F1=nothing, proxdiv_F2=nothing, kwargs...
)
    if proxdiv_F1 !== nothing && proxdiv_F2 !== nothing
        Base.depwarn(
            "keyword arguments `proxdiv_F1` and `proxdiv_F2` are deprecated",
            :sinkhorn_unbalanced,
        )

        # have to wrap the "proxdiv" functions since the signature changed
        # ε was fixed in the function, so we ignore it
        proxdiv_F1_wrapper(s, p, _) = copyto!(s, proxdiv_F1(s, p))
        proxdiv_F2_wrapper(s, p, _) = copyto!(s, proxdiv_F2(s, p))

        return sinkhorn_unbalanced(
            μ, ν, C, proxdiv_F1_wrapper, proxdiv_F2_wrapper, ε; kwargs...
        )
    end

    # define "proxdiv" functions for the unbalanced OT problem
    proxdivF!(s, p, ε, λ) = (s .= (p ./ s) .^ (λ / (ε + λ)))
    proxdivF1!(s, p, ε) = proxdivF!(s, p, ε, λ1)
    proxdivF2!(s, p, ε) = proxdivF!(s, p, ε, λ2)

    return sinkhorn_unbalanced(μ, ν, C, proxdivF1!, proxdivF2!, ε; kwargs...)
end

"""
    sinkhorn_unbalanced(
        μ, ν, C, proxdivF1!, proxdivF2!, ε;
        atol=0, rtol=atol > 0 ? 0 : √eps, check_convergence=10, maxiter=1_000,
    )

Compute the optimal transport plan for the unbalanced entropically regularized optimal
transport problem with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, entropic regularization parameter `ε`, and soft marginal
constraints ``F_1`` and ``F_2`` with "proxdiv" functions `proxdivF!` and `proxdivG!`.

The optimal transport plan `γ` is of the same size as `C` and solves
```math
\\inf_{\\gamma} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma)
+ F_1(\\gamma 1, \\mu)
+ F_2(\\gamma^{\\mathsf{T}} 1, \\nu),
```
where ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{i,j} \\log \\gamma_{i,j}`` is the entropic
regularization term and ``F_1(\\cdot, \\mu)`` and ``F_2(\\cdot, \\nu)`` are soft marginal
constraints for the source and target marginals.

The functions `proxdivF1!(s, p, ε)` and `proxdivF2!(s, p, ε)` evaluate the "proxdiv"
functions of ``F_1(\\cdot, p)`` and ``F_2(\\cdot, p)`` at ``s`` for the entropic
regularization parameter ``\\varepsilon``. They have to be mutating and overwrite the first
argument `s` with the result of their computations.

Mathematically, the "proxdiv" functions are defined as
```math
\\operatorname{proxdiv}_{F_i}(s, p, \\varepsilon)
= \\operatorname{prox}^{\\operatorname{KL}}_{F_i(\\cdot, p)/\\varepsilon}(s) \\oslash s
```
where ``\\oslash`` denotes element-wise division and
``\\operatorname{prox}_{F_i(\\cdot, p)/\\varepsilon}^{\\operatorname{KL}}`` is the proximal
operator of ``F_i(\\cdot, p)/\\varepsilon`` for the Kullback-Leibler
(``\\operatorname{KL}``) divergence.  It is defined as
```math
\\operatorname{prox}_{F}^{\\operatorname{KL}}(x)
= \\operatorname{argmin}_{y} F(y) + \\operatorname{KL}(y|x)
```
and can be computed in closed-form for specific choices of ``F``. For instance, if
``F(\\cdot, p) = \\lambda \\operatorname{KL}(\\cdot | p)`` (``\\lambda > 0``), then
```math
\\operatorname{prox}_{F(\\cdot, p)/\\varepsilon}^{\\operatorname{KL}}(x)
= x^{\\frac{\\varepsilon}{\\varepsilon + \\lambda}} p^{\\frac{\\lambda}{\\varepsilon + \\lambda}},
```
where all operators are acting pointwise.[^CPSV18]

Every `check_convergence` steps it is assessed if the algorithm is converged by checking if
the iterates of the scaling factor in the current and previous iteration satisfy
`isapprox(vcat(a, b), vcat(aprev, bprev); atol=atol, rtol=rtol)` where `a` and `b` are the
current iterates and `aprev` and `bprev` the previous ones. The default `rtol` depends on
the types of `μ`, `ν`, and `C`. After `maxiter` iterations, the computation is stopped.

[^CPSV18]: Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F.-X. (2018). [Scaling algorithms for unbalanced optimal transport problems](https://doi.org/10.1090/mcom/3303). Mathematics of Computation, 87(314), 2563–2609.

See also: [`sinkhorn_unbalanced2`](@ref)
"""
function sinkhorn_unbalanced(
    μ,
    ν,
    C,
    proxdivF1!,
    proxdivF2!,
    ε;
    tol=nothing,
    atol=tol,
    rtol=nothing,
    max_iter=nothing,
    maxiter=max_iter,
    check_convergence::Int=10,
)
    # deprecations
    if tol !== nothing
        Base.depwarn(
            "keyword argument `tol` is deprecated, please use `atol` and `rtol`",
            :sinkhorn_unbalanced,
        )
    end
    if max_iter !== nothing
        Base.depwarn(
            "keyword argument `max_iter` is deprecated, please use `maxiter`",
            :sinkhorn_unbalanced,
        )
    end

    # compute Gibbs kernel
    K = @. exp(-C / ε)

    # set default values of squared tolerances
    T = float(Base.promote_eltype(μ, ν, K))
    sqatol = atol === nothing ? 0 : atol^2
    sqrtol = rtol === nothing ? (sqatol > zero(sqatol) ? zero(T) : eps(T)) : rtol^2

    # initialize iterates
    a = similar(μ, T)
    sum!(a, K)
    proxdivF1!(a, μ, ε)
    b = similar(ν, T)
    mul!(b, K', a)
    proxdivF2!(b, ν, ε)

    # caches for convergence checks
    a_old = similar(a)
    b_old = similar(b)

    isconverged = false
    _maxiter = maxiter === nothing ? 1_000 : maxiter
    for iter in 1:_maxiter
        # update cache if necessary
        ischeck = iter % check_convergence == 0
        if ischeck
            copyto!(a_old, a)
            copyto!(b_old, b)
        end

        # compute next iterates
        mul!(a, K, b)
        proxdivF1!(a, μ, ε)
        mul!(b, K', a)
        proxdivF2!(b, ν, ε)

        # check convergence of the scaling factors
        if ischeck
            # compute norm of current and previous scaling factors and their difference
            sqnorm_a_b = sum(abs2, a) + sum(abs2, b)
            sqnorm_a_b_old = sum(abs2, a_old) + sum(abs2, b_old)
            a_old .-= a
            b_old .-= b
            sqeuclidean_a_b = sum(abs2, a_old) + sum(abs2, b_old)
            @debug "Sinkhorn algorithm (" *
                   string(iter) *
                   "/" *
                   string(_maxiter) *
                   ": squared Euclidean distance of iterates = " *
                   string(sqeuclidean_a_b)

            # check convergence of `a`
            if sqeuclidean_a_b < max(sqatol, sqrtol * max(sqnorm_a_b, sqnorm_a_b_old))
                @debug "Sinkhorn algorithm ($iter/$_maxiter): converged"
                isconverged = true
                break
            end
        end
    end

    if !isconverged
        @warn "Sinkhorn algorithm ($_maxiter/$_maxiter): not converged"
    end

    return K .* a .* b'
end

"""
    sinkhorn_unbalanced2(μ, ν, C, λ1, λ2, ε; plan=nothing, kwargs...)
    sinkhorn_unbalanced2(μ, ν, C, proxdivF1!, proxdivF2!, ε; plan=nothing, kwargs...)

Compute the optimal transport plan for the unbalanced entropically regularized optimal
transport problem with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, entropic regularization parameter `ε`, and marginal relaxation
terms `λ1` and `λ2` or soft marginal constraints with "proxdiv" functions `proxdivF1!` and
`proxdivF2!`.

A pre-computed optimal transport `plan` may be provided. The other keyword arguments
supported here are the same as those in the [`sinkhorn_unbalanced`](@ref) for unbalanced
optimal transport problems with general soft marginal constraints.

See also: [`sinkhorn_unbalanced`](@ref)
"""
function sinkhorn_unbalanced2(
    μ, ν, C, λ1_or_proxdivF1, λ2_or_proxdivF2, ε; plan=nothing, kwargs...
)
    γ = if plan === nothing
        sinkhorn_unbalanced(μ, ν, C, λ1_or_proxdivF1, λ2_or_proxdivF2, ε; kwargs...)
    else
        # check dimensions
        size(C) == (length(μ), length(ν)) ||
            error("cost matrix `C` must be of size `(length(μ), length(ν))`")
        size(plan) == size(C) || error(
            "optimal transport plan `plan` and cost matrix `C` must be of the same size",
        )
        plan
    end
    return dot(γ, C)
end

"""
    sinkhorn_barycenter(μ, C, ε, w; tol=1e-9, check_marginal_step=10, max_iter=1000)

Compute the Sinkhorn barycenter for a collection of `N` histograms contained in the columns of `μ`, for a cost matrix `C` of size `(size(μ, 1), size(μ, 1))`, relative weights `w` of size `N`, and entropic regularisation parameter `ε`.
Returns the entropically regularised barycenter of the `μ`, i.e. the histogram `ρ` of length `size(μ, 1)` that solves

```math
\\min_{\\rho \\in \\Sigma} \\sum_{i = 1}^N w_i \\operatorname{OT}_{\\varepsilon}(\\mu_i, \\rho)
```

where ``\\operatorname{OT}_{ε}(\\mu, \\nu) = \\inf_{\\gamma \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle + \\varepsilon \\Omega(\\gamma)``
is the entropic optimal transport loss with cost ``C`` and regularisation ``\\epsilon``.
"""
function sinkhorn_barycenter(μ, C, ε, w; tol=1e-9, check_marginal_step=10, max_iter=1000)
    sums = sum(μ; dims=1)
    if !isapprox(extrema(sums)...)
        throw(ArgumentError("Error: marginals are unbalanced"))
    end
    K = exp.(-C / ε)
    converged = false
    v = ones(size(μ))
    u = ones(size(μ))
    N = size(μ, 2)
    for n in 1:max_iter
        v = μ ./ (K' * u)
        a = ones(size(u, 1))
        a = prod((K * v)' .^ w; dims=1)'
        u = a ./ (K * v)
        if n % check_marginal_step == 0
            # check marginal errors
            err = maximum(abs.(μ .- v .* (K' * u)))
            @debug "Sinkhorn algorithm: iteration $n" err
            if err < tol
                converged = true
                break
            end
        end
    end
    if !converged
        @warn "Sinkhorn did not converge"
    end
    return u[:, 1] .* (K * v[:, 1])
end
