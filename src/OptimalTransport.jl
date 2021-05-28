# OptimalTransport.jl -- optimal transportation algorithms for Julia
# See prettyprinted documentation at https://juliaoptimaltransport.github.io/OptimalTransport.jl/dev/
#

module OptimalTransport

using Distances
using LinearAlgebra
using IterativeSolvers, SparseArrays
using LogExpFunctions: LogExpFunctions
using MathOptInterface
using Distributions
using PDMats
using QuadGK

export sinkhorn, sinkhorn2
export emd, emd2
export sinkhorn_stabilized, sinkhorn_stabilized_epsscaling, sinkhorn_barycenter
export sinkhorn_unbalanced, sinkhorn_unbalanced2
export quadreg
export ot_cost, ot_plan

const MOI = MathOptInterface

include("distances/bures.jl")

"""
    emd(μ, ν, C, optimizer)

Compute the optimal transport plan `γ` for the Monge-Kantorovich problem with source
histogram `μ`, target histogram `ν`, and cost matrix `C` of size `(length(μ), length(ν))`
which solves
```math
\\inf_{γ ∈ Π(μ, ν)} \\langle γ, C \\rangle.
```

The corresponding linear programming problem is solved with the user-provided `optimizer`.
Possible choices are `Tulip.Optimizer()` and `Clp.Optimizer()` in the `Tulip` and `Clp`
packages, respectively.
"""
function emd(μ, ν, C, model::MOI.ModelLike)
    # check size of cost matrix
    nμ = length(μ)
    nν = length(ν)
    size(C) == (nμ, nν) || error("cost matrix `C` must be of size `(length(μ), length(ν))`")
    nC = length(C)

    # define variables
    x = MOI.add_variables(model, nC)
    xmat = reshape(x, nμ, nν)

    # define objective function
    T = float(eltype(C))
    zero_T = zero(T)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(float.(vec(C)), x), zero_T),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # add non-negativity constraints
    for xi in x
        MOI.add_constraint(model, MOI.SingleVariable(xi), MOI.GreaterThan(zero_T))
    end

    # add constraints for source
    for (i, μi) in zip(axes(xmat, 1), μ) # eachrow(xmat) is not available on Julia 1.0
        f = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(one(μi), xi) for xi in view(xmat, i, :)], zero(μi)
        )
        MOI.add_constraint(model, f, MOI.EqualTo(μi))
    end

    # add constraints for target
    for (i, νi) in zip(axes(xmat, 2), ν) # eachcol(xmat) is not available on Julia 1.0
        f = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(one(νi), xi) for xi in view(xmat, :, i)], zero(νi)
        )
        MOI.add_constraint(model, f, MOI.EqualTo(νi))
    end

    # compute optimal solution
    MOI.optimize!(model)
    status = MOI.get(model, MOI.TerminationStatus())
    status === MOI.OPTIMAL || error("failed to compute optimal transport plan: ", status)
    p = MOI.get(model, MOI.VariablePrimal(), x)
    γ = reshape(p, nμ, nν)

    return γ
end

"""
    emd2(μ, ν, C, optimizer; plan=nothing)

Compute the optimal transport cost (a scalar) for the Monge-Kantorovich problem with source
histogram `μ`, target histogram `ν`, and cost matrix `C` of size `(length(μ), length(ν))`
which is given by
```math
\\inf_{γ ∈ Π(μ, ν)} \\langle γ, C \\rangle.
```

The corresponding linear programming problem is solved with the user-provided `optimizer`.
Possible choices are `Tulip.Optimizer()` and `Clp.Optimizer()` in the `Tulip` and `Clp`
packages, respectively.

A pre-computed optimal transport `plan` may be provided.
"""
function emd2(μ, ν, C, optimizer; plan=nothing)
    γ = if plan === nothing
        # compute optimal transport plan
        emd(μ, ν, C, optimizer)
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
    sum(μ) ≈ sum(ν) ||
        throw(ArgumentError("source and target marginals must have the same mass"))

    # set default values of tolerances
    T = float(Base.promote_eltype(μ, ν, K))
    _atol = atol === nothing ? 0 : atol
    _rtol = rtol === nothing ? (_atol > zero(_atol) ? zero(T) : sqrt(eps(T))) : rtol

    # initial iteration
    u = μ ./ sum(K; dims=2)
    v = ν ./ (K' * u)
    tmp1 = K * v
    tmp2 = similar(u)

    norm_μ = sum(abs, μ) # for convergence check
    isconverged = false
    check_step = check_convergence === nothing ? 10 : check_convergence
    for iter in 0:maxiter
        if iter % check_step == 0
            # check source marginal
            # do not overwrite `tmp1` but reuse it for computing `u` if not converged
            @. tmp2 = u * tmp1
            norm_uKv = sum(abs, tmp2)
            @. tmp2 = μ - tmp2
            norm_diff = sum(abs, tmp2)

            @debug "Sinkhorn algorithm (" *
                   string(iter) *
                   "/" *
                   string(maxiter) *
                   ": absolute error of source marginal = " *
                   string(norm_diff)

            # check stopping criterion
            if norm_diff < max(_atol, _rtol * max(norm_μ, norm_uKv))
                @debug "Sinkhorn algorithm ($iter/$maxiter): converged"
                isconverged = true
                break
            end
        end

        # perform next iteration
        if iter < maxiter
            @. u = μ / tmp1
            mul!(v, K', u)
            @. v = ν / v
            mul!(tmp1, K, v)
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

See also: [`sinkhorn2`](@ref)
"""
function sinkhorn(μ, ν, C, ε; kwargs...)
    # compute Gibbs kernel
    K = @. exp(-C / ε)

    # compute dual potentials
    u, v = sinkhorn_gibbs(μ, ν, K; kwargs...)

    return K .* u .* v'
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
        size(C) == (length(μ), length(ν)) ||
            error("cost matrix `C` must be of size `(length(μ), length(ν))`")
        size(plan) == size(C) || error(
            "optimal transport plan `plan` and cost matrix `C` must be of the same size",
        )
        plan
    end

    cost = if regularization
        dot(γ, C) + ε * sum(LogExpFunctions.xlogx, γ)
    else
        dot(γ, C)
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
    sinkhorn_stabilized_epsscaling(mu, nu, C, eps; absorb_tol = 1e3, max_iter = 1000, tol = 1e-9, lambda = 0.5, k = 5, verbose = false)

Compute optimal transport plan of histograms `mu` and `nu` with cost matrix `C` and entropic regularisation parameter `eps`. 
Uses stabilized Sinkhorn algorithm with epsilon-scaling (Schmitzer et al., 2019). 

`k` epsilon-scaling steps are used with scaling factor `lambda`, i.e. sequentially solve Sinkhorn with regularisation parameters 
`[lambda^(1-k), ..., lambda^(-1), 1]*eps`. 
"""
function sinkhorn_stabilized_epsscaling(
    mu, nu, C, eps; absorb_tol=1e3, max_iter=1000, tol=1e-9, lambda=0.5, k=5, verbose=false
)
    eps_values = [eps * lambda^(k - j) for j in 1:k]
    alpha = zeros(size(mu))
    beta = zeros(size(nu))
    for eps in eps_values
        if verbose
            println(string("Warm start: eps = ", eps))
        end
        alpha, beta = sinkhorn_stabilized(
            mu,
            nu,
            C,
            eps;
            absorb_tol=absorb_tol,
            max_iter=max_iter,
            tol=tol,
            alpha=alpha,
            beta=beta,
            return_duals=true,
            verbose=verbose,
        )
    end
    K = exp.(-(C .- alpha .- beta') / eps) .* mu .* nu'
    return K
end

function getK(C, alpha, beta, eps, mu, nu)
    return (exp.(-(C .- alpha .- beta') / eps) .* mu .* nu')
end

"""
    sinkhorn_stabilized(mu, nu, C, eps; absorb_tol = 1e3, max_iter = 1000, tol = 1e-9, alpha = nothing, beta = nothing, return_duals = false, verbose = false)

Compute optimal transport plan of histograms `mu` and `nu` with cost matrix `C` and entropic regularisation parameter `eps`. 
Uses stabilized Sinkhorn algorithm (Schmitzer et al., 2019).
"""
function sinkhorn_stabilized(
    mu,
    nu,
    C,
    eps;
    absorb_tol=1e3,
    max_iter=1000,
    tol=1e-9,
    alpha=zeros(size(mu)),
    beta=zeros(size(nu)),
    return_duals=false,
    verbose=false,
)
    u = ones(size(mu))
    v = ones(size(nu))
    K = getK(C, alpha, beta, eps, mu, nu)
    i = 0

    if !(sum(mu) ≈ sum(nu))
        throw(ArgumentError("Error: mu and nu must lie in the simplex"))
    end

    while true
        u = mu ./ (K * v .+ 1e-16)
        v = nu ./ (K' * u .+ 1e-16)
        if (max(norm(u, Inf), norm(v, Inf)) > absorb_tol)
            if verbose
                println("Absorbing (u, v) into (alpha, beta)")
            end
            # absorb into α, β
            alpha = alpha + eps * log.(u)
            beta = beta + eps * log.(v)
            u = ones(size(mu))
            v = ones(size(nu))
            K = getK(C, alpha, beta, eps, mu, nu)
        end
        if i % 10 == 0
            # check marginal
            gamma = getK(C, alpha, beta, eps, mu, nu) .* (u .* v')
            err_mu = norm(gamma * ones(size(nu)) - mu, Inf)
            err_nu = norm(gamma' * ones(size(mu)) - nu, Inf)
            if verbose
                println(string("Iteration ", i, ", err = ", 0.5 * (err_mu + err_nu)))
            end
            if 0.5 * (err_mu + err_nu) < tol
                break
            end
        end

        if i > max_iter
            if verbose
                println("Warning: exited before convergence")
            end
            break
        end
        i += 1
    end
    alpha = alpha + eps * log.(u)
    beta = beta + eps * log.(v)
    if return_duals
        return alpha, beta
    end
    return getK(C, alpha, beta, eps, mu, nu)
end

"""
    sinkhorn_barycenter(mu_all, C_all, eps, lambda_all; tol = 1e-9, check_marginal_step = 10, max_iter = 1000)

Compute the entropically regularised (i.e. Sinkhorn) barycenter for a collection of `N`
histograms `mu_all` with respective cost matrices `C_all`, relative weights `lambda_all`,
and entropic regularisation parameter `eps`. 

 - `mu_all` is taken to contain `N` histograms `mu_all[i, :]` for `math i = 1, \\ldots, N`.
 - `C_all` is taken to be a list of `N` cost matrices corresponding to the `mu_all[i, :]`.
 - `eps` is the scalar regularisation parameter.
 - `lambda_all` are positive weights.

Returns the entropically regularised barycenter of the `mu_all`, i.e. the distribution that minimises

```math
\\min_{\\mu \\in \\Sigma} \\sum_{i = 1}^N \\lambda_i \\mathrm{entropicOT}^{\\epsilon}_{C_i}(\\mu, \\mu_i)
```

where ``\\mathrm{entropicOT}^{\\epsilon}_{C}`` denotes the entropic optimal transport cost with cost ``C`` and entropic regularisation level ``\\epsilon``.
"""
function sinkhorn_barycenter(
    mu_all, C_all, eps, lambda_all; tol=1e-9, check_marginal_step=10, max_iter=1000
)
    sums = sum(mu_all; dims=2)
    if !isapprox(extrema(sums)...)
        throw(ArgumentError("Error: marginals are unbalanced"))
    end
    K_all = [exp.(-C_all[i] / eps) for i in 1:length(C_all)]
    converged = false
    v_all = ones(size(mu_all))
    u_all = ones(size(mu_all))
    N = size(mu_all, 1)
    for n in 1:max_iter
        for i in 1:N
            v_all[i, :] = mu_all[i, :] ./ (K_all[i]' * u_all[i, :])
        end
        a = ones(size(u_all, 2))
        for i in 1:N
            a = a .* ((K_all[i] * v_all[i, :]) .^ (lambda_all[i]))
        end
        for i in 1:N
            u_all[i, :] = a ./ (K_all[i] * v_all[i, :])
        end
        if n % check_marginal_step == 0
            # check marginal errors
            err = maximum([
                maximum(abs.(mu_all[i, :] .- v_all[i, :] .* (K_all[i]' * u_all[i, :]))) for
                i in 1:N
            ])
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
    return u_all[1, :] .* (K_all[1] * v_all[1, :])
end

"""
    quadreg(mu, nu, C, ϵ; θ = 0.1, tol = 1e-5,maxiter = 50,κ = 0.5,δ = 1e-5)

Computes the optimal transport plan of histograms `mu` and `nu` with cost matrix `C` and quadratic regularization parameter `ϵ`, 
using the semismooth Newton algorithm [Lorenz 2016].

This implementation makes use of IterativeSolvers.jl and SparseArrays.jl.

Parameters:\n
θ: starting Armijo parameter.\n
tol: tolerance of marginal error.\n
maxiter: maximum interation number.\n
κ: control parameter of Armijo.\n
δ: small constant for the numerical stability of conjugate gradient iterative solver.\n

Tips:
If the algorithm does not converge, try some different values of θ.

Reference:
Lorenz, D.A., Manns, P. and Meyer, C., 2019. Quadratically regularized optimal transport. arXiv preprint arXiv:1903.01112v4.
"""
function quadreg(mu, nu, C, ϵ; θ=0.1, tol=1e-5, maxiter=50, κ=0.5, δ=1e-5)
    if !(sum(mu) ≈ sum(nu))
        throw(ArgumentError("Error: mu and nu must lie in the simplex"))
    end

    N = length(mu)
    M = length(nu)

    # initialize dual potentials as uniforms
    a = ones(M) ./ M
    b = ones(N) ./ N
    γ = spzeros(M, N)

    da = spzeros(M)
    db = spzeros(N)

    converged = false

    function DualObjective(a, b)
        A = a .* ones(N)' + ones(M) .* b' - C'

        return 0.5 * norm(A[A .> 0], 2)^2 - ϵ * (dot(nu, a) + dot(mu, b))
    end

    # computes minimizing directions, update γ
    function search_dir!(a, b, da, db)
        P = a * ones(N)' .+ ones(M) * b' .- C'

        σ = 1.0 * sparse(P .>= 0)
        γ = sparse(max.(P, 0) ./ ϵ)

        G = vcat(
            hcat(spdiagm(0 => σ * ones(N)), σ),
            hcat(sparse(σ'), spdiagm(0 => sparse(σ') * ones(M))),
        )

        h = vcat(γ * ones(N) - nu, sparse(γ') * ones(M) - mu)

        x = cg(G + δ * I, -ϵ * h)

        da = x[1:M]
        return db = x[(M + 1):end]
    end

    function search_dir(a, b)
        P = a * ones(N)' .+ ones(M) * b' .- C'

        σ = 1.0 * sparse(P .>= 0)
        γ = sparse(max.(P, 0) ./ ϵ)

        G = vcat(
            hcat(spdiagm(0 => σ * ones(N)), σ),
            hcat(sparse(σ'), spdiagm(0 => sparse(σ') * ones(M))),
        )

        h = vcat(γ * ones(N) - nu, sparse(γ') * ones(M) - mu)

        x = cg(G + δ * I, -ϵ * h)

        return x[1:M], x[(M + 1):end]
    end

    # computes optimal maginitude in the minimizing directions
    function search_t(a, b, da, db, θ)
        d = ϵ * dot(γ, (da .* ones(N)' .+ ones(M) .* db')) - ϵ * (dot(da, nu) + dot(db, mu))

        ϕ₀ = DualObjective(a, b)
        t = 1

        while DualObjective(a + t * da, b + t * db) >= ϕ₀ + t * θ * d
            t *= κ

            if t < 1e-15
                # @warn "@ i = $i, t = $t , armijo did not converge"
                break
            end
        end
        return t
    end

    for i in 1:maxiter

        # search_dir!(a, b, da, db)
        da, db = search_dir(a, b)

        t = search_t(a, b, da, db, θ)

        a += t * da
        b += t * db

        err1 = norm(γ * ones(N) - nu, Inf)
        err2 = norm(sparse(γ') * ones(M) - mu, Inf)

        if err1 <= tol && err2 <= tol
            converged = true
            @warn "Converged @ i = $i with marginal errors: \n err1 = $err1, err2 = $err2 \n"
            break
        elseif i == maxiter
            @warn "Not Converged with errors:\n err1 = $err1, err2 = $err2 \n"
        end

        @debug " t = $t"
        @debug "marginal @ i = $i: err1 = $err1, err2 = $err2 "
    end

    if !converged
        @warn "SemiSmooth Newton algorithm did not converge"
    end

    return sparse(γ')
end

"""
    ot_cost(
        c, μ::ContinuousUnivariateDistribution, ν::UnivariateDistribution; plan=nothing
    )

Compute the optimal transport cost for the Monge-Kantorovich problem with univariate
distributions `μ` and `ν` as source and target marginals and cost function `c` of
the form ``c(x, y) = h(|x - y|)`` where ``h`` is a convex function.

In this setting, the optimal transport cost can be computed as
```math
\\int_0^1 c(F_\\mu^{-1}(x), F_\\nu^{-1}(x)) \\mathrm{d}x
```
where ``F_\\mu^{-1}`` and ``F_\\nu^{-1}`` are the quantile functions of `μ` and `ν`,
respectively.

A pre-computed optimal transport `plan` may be provided.

See also: [`ot_plan`](@ref), [`emd2`](@ref)
"""
function ot_cost(
    c, μ::ContinuousUnivariateDistribution, ν::UnivariateDistribution; plan=nothing
)
    cost, _ = if plan === nothing
        quadgk(0, 1) do q
            return c(quantile(μ, q), quantile(ν, q))
        end
    else
        quadgk(0, 1) do q
            x = quantile(μ, q)
            return c(x, plan(x))
        end
    end
    return cost
end

"""
    ot_plan(c, μ::ContinuousUnivariateDistribution, ν::UnivariateDistribution)

Compute the optimal transport plan for the Monge-Kantorovich problem with univariate
distributions `μ` and `ν` as source and target marginals and cost function `c` of
the form ``c(x, y) = h(|x - y|)`` where ``h`` is a convex function.

In this setting, the optimal transport plan is the Monge map
```math
T = F_\\nu^{-1} \\circ F_\\mu
```
where ``F_\\mu`` is the cumulative distribution function of `μ` and ``F_\\nu^{-1}`` is the
quantile function of `ν`.

See also: [`ot_cost`](@ref), [`emd`](@ref)
"""
function ot_plan(c, μ::ContinuousUnivariateDistribution, ν::UnivariateDistribution)
    # Use T instead of γ to indicate that this is a Monge map.
    T(x) = quantile(ν, cdf(μ, x))
    return T
end

"""
    ot_cost(c,mu::MvNormal,nu::MvNormal; plan=nothing)
Compute the optimal transport cost between μ to ν, where
both are Normal or Multivariate Normal distributions and the cost
function ``c(x,y) = (||x-y||)^2``

Return optimal transport cost.
"""
function ot_cost(
    c::SqEuclidean, μ::MvNormal, ν::MvNormal; plan=nothing
)
    return sqeuclidean(μ.μ, ν.μ) + sqbures(μ.Σ, ν.Σ)
end


"""
    _gaussian_ot_A(A::AbstractMatrix, B::AbstractMatrix)
Compute
```math
\\Sigma_\\mu^{-1/2}
(
\\Sigma_\\mu^{1/2}
\\Sigma_\\nu
\\Sigma_\\mu^{1/2}
)
\\Sigma_\\mu^{-1/2}
```
"""
function _gaussian_ot_A(A::AbstractMatrix, B::AbstractMatrix)
    sqrt_A = sqrt(A)
    return sqrt_A * B * sqrt_A
end
function _gaussian_ot_A(A::PDMats.PDiagMat, B::AbstractMatrix)
    return sqrt.(A.diag) .* B .* sqrt.(A.diag')
end
function _gaussian_ot_A(A::StridedMatrix, B::PDMats.PDMat)
    return PDMats.X_A_Xt(B, sqrt(A))
end
_gaussian_ot_A(A::PDMats.PDMat, B::PDMats.PDMat) = _gaussian_ot_A(A.mat, B)
_gaussian_ot_A(A::AbstractMatrix, B::PDMats.PDiagMat) = _gaussian_ot_A(B, A)
_gaussian_ot_A(A::PDMats.PDMat, B::StridedMatrix) = _gaussian_ot_A(B, A)


"""
    ot_plan(c,mu::MvNormal,nu::MvNormal; plan=nothing)
Compute the optimal transport plan between μ to ν, where
both are Normal or Multivariate Normal distributions and the cost
function ``c(x,y) = (||x-y||)^2``

Return optimal transport plan.
"""
function ot_plan(
    c::SqEuclidean, μ::MvNormal, ν::MvNormal
)
    Σμsqrt = μ.Σ^(-1/2)
    T(x)   = ν.μ + (Σμsqrt*sqrt(_gaussian_ot_A(μ.Σ,ν.Σ))*Σμsqrt)*(x - μ.μ)
    return T
end


end