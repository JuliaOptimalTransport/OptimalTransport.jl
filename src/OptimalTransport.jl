# OptimalTransport.jl -- optimal transportation algorithms for Julia
# See prettyprinted documentation at https://juliaoptimaltransport.github.io/OptimalTransport.jl/dev/
#

module OptimalTransport

using Distances
using LinearAlgebra
using IterativeSolvers, SparseArrays
using Requires
using MathOptInterface
using Distributions, QuadGK
using StatsBase

export sinkhorn, sinkhorn2
export emd, emd2
export sinkhorn_stabilized, sinkhorn_stabilized_epsscaling, sinkhorn_barycenter
export sinkhorn_unbalanced, sinkhorn_unbalanced2
export quadreg
export otcost, otplan

const MOI = MathOptInterface

function __init__()
    @require PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0" begin
        export POT
        include("pot.jl")
    end
end

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
    T = eltype(C)
    zero_T = zero(T)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(vec(C), x), zero_T),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # add non-negativity constraints
    for xi in x
        MOI.add_constraint(model, MOI.SingleVariable(xi), MOI.GreaterThan(zero_T))
    end

    # add constraints for source
    for (xs, μi) in zip(eachrow(xmat), μ)
        f = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(one(μi), xi) for xi in xs], zero(μi)
        )
        MOI.add_constraint(model, f, MOI.EqualTo(μi))
    end

    # add constraints for target
    for (xs, νi) in zip(eachcol(xmat), ν)
        f = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(one(νi), xi) for xi in xs], zero(νi)
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
function emd2(μ, ν, C, optimizer; map=nothing, plan=map)
    # check deprecation
    if map !== nothing
        Base.depwarn("the keyword argument `map` is deprecated, please use `plan`", :emd2)
    end

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
    sinkhorn_gibbs(mu, nu, K; tol=1e-9, check_marginal_step=10, maxiter=1000)

Compute dual potentials `u` and `v` for histograms `mu` and `nu` and Gibbs kernel `K` using
the Sinkhorn algorithm (Peyre et al., 2019)

The Gibbs kernel `K` is given by `K = exp.(- C / eps)` where `C` is the cost matrix and
`eps` the entropic regularization parameter. The optimal transport plan for histograms `u`
and `v` and cost matrix `C` with regularization parameter `eps` can be computed as
`Diagonal(u) * K * Diagonal(v)`.
"""
function sinkhorn_gibbs(mu, nu, K; tol=1e-9, check_marginal_step=10, maxiter=1000)
    if !(sum(mu) ≈ sum(nu))
        throw(ArgumentError("Error: mu and nu must lie in the simplex"))
    end

    # initial iteration
    temp_v = vec(sum(K; dims=2))
    u = mu ./ temp_v
    temp_u = K' * u
    v = nu ./ temp_u

    isconverged = false
    for iter in 0:maxiter
        # check mu marginal
        if iter % check_marginal_step == 0
            mul!(temp_v, K, v)
            @. temp_v = abs(mu - u * temp_v)

            err = maximum(temp_v)
            @debug "Sinkhorn algorithm: iteration $iter" err

            # check stopping criterion
            if err < tol
                isconverged = true
                break
            end
        end

        # perform next iteration
        if iter < maxiter
            mul!(temp_v, K, v)
            @. u = mu / temp_v
            mul!(temp_u, K', u)
            @. v = nu / temp_u
        end
    end

    if !isconverged
        @warn "Sinkhorn algorithm did not converge"
    end

    return u, v
end

"""
    sinkhorn(mu, nu, C, eps; tol=1e-9, check_marginal_step=10, maxiter=1000)

Compute entropically regularised transport plan of histograms `mu` and `nu` with cost matrix `C` and entropic
regularization parameter `eps`. 

Return optimal transport coupling `γ` of the same dimensions as `C` which solves 

```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle - \\epsilon H(\\gamma)
```

where ``H`` is the entropic regulariser, ``H(\\gamma) = -\\sum_{i, j} \\gamma_{ij} \\log(\\gamma_{ij})``.

"""
function sinkhorn(mu, nu, C, eps; kwargs...)
    # compute Gibbs kernel
    K = @. exp(-C / eps)

    # compute dual potentials
    u, v = sinkhorn_gibbs(mu, nu, K; kwargs...)

    return Diagonal(u) * K * Diagonal(v)
end

"""
    sinkhorn2(mu, nu, C, eps; plan=nothing, kwargs...)

Compute entropically regularised transport cost of histograms `mu` and `nu` with cost matrix `C` and entropic
regularization parameter `eps`.

Return optimal value of

```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle - \\epsilon H(\\gamma)
```

where ``H`` is the entropic regulariser, ``H(\\gamma) = -\\sum_{i, j} \\gamma_{ij} \\log(\\gamma_{ij})``.

A pre-computed optimal transport `plan` may be provided.

See also: [`sinkhorn`](@ref)
"""
function sinkhorn2(μ, ν, C, ε; map=nothing, plan=map, kwargs...)
    # check deprecation
    if map !== nothing
        Base.depwarn(
            "the keyword argument `map` is deprecated, please use `plan`", :sinkhorn2
        )
    end

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
    return dot(γ, C)
end

"""
    sinkhorn_unbalanced(mu, nu, C, lambda1, lambda2, eps; tol = 1e-9, max_iter = 1000, verbose = false, proxdiv_F1 = nothing, proxdiv_F2 = nothing)

Computes the optimal transport plan of histograms `mu` and `nu` with cost matrix `C` and entropic regularization parameter `eps`, 
using the unbalanced Sinkhorn algorithm [Chizat 2016] with KL-divergence terms for soft marginal constraints, with weights `(lambda1, lambda2)`
for the marginals `mu`, `nu` respectively.

For full generality, the user can specify the soft marginal constraints ``(F_1(\\cdot | \\mu), F_2(\\cdot | \\nu))`` to the problem

```math
\\min_\\gamma \\epsilon \\mathrm{KL}(\\gamma | \\exp(-C/\\epsilon)) + F_1(\\gamma_1 | \\mu) + F_2(\\gamma_2 | \\nu)
```

via `math\\mathrm{proxdiv}_{F_1}(s, p)` and `math\\mathrm{proxdiv}_{F_2}(s, p)` (see Chizat et al., 2016 for details on this). If specified, the algorithm will use the user-specified F1, F2 rather than the default (a KL-divergence).
"""
function sinkhorn_unbalanced(
    mu,
    nu,
    C,
    lambda1,
    lambda2,
    eps;
    tol=1e-9,
    max_iter=1000,
    verbose=false,
    proxdiv_F1=nothing,
    proxdiv_F2=nothing,
)
    function proxdiv_KL(s, eps, lambda, p)
        return @. (s^(eps / (eps + lambda)) * p^(lambda / (eps + lambda))) / s
    end

    a = ones(size(mu, 1))
    b = ones(size(nu, 1))
    a_old = a
    b_old = b
    tmp_a = zeros(size(nu, 1))
    tmp_b = zeros(size(mu, 1))

    K = @. exp(-C / eps)

    iter = 1

    while true
        a_old = a
        b_old = b
        tmp_b = K * b
        if proxdiv_F1 == nothing
            a = proxdiv_KL(tmp_b, eps, lambda1, mu)
        else
            a = proxdiv_F1(tmp_b, mu)
        end
        tmp_a = K' * a
        if proxdiv_F2 == nothing
            b = proxdiv_KL(tmp_a, eps, lambda2, nu)
        else
            b = proxdiv_F2(tmp_a, nu)
        end
        iter += 1
        if iter % 10 == 0
            err_a =
                maximum(abs.(a - a_old)) / max(maximum(abs.(a)), maximum(abs.(a_old)), 1)
            err_b =
                maximum(abs.(b - b_old)) / max(maximum(abs.(b)), maximum(abs.(b_old)), 1)
            if verbose
                println("Iteration $iter, err = ", 0.5 * (err_a + err_b))
            end
            if (0.5 * (err_a + err_b) < tol) || iter > max_iter
                break
            end
        end
    end
    if iter > max_iter && verbose
        println("Warning: exited before convergence")
    end
    return Diagonal(a) * K * Diagonal(b)
end

"""
    sinkhorn_unbalanced2(mu, nu, C, lambda1, lambda2, eps; plan=nothing, kwargs...)

Computes the optimal transport cost of histograms `mu` and `nu` with cost matrix `C` and entropic regularization parameter `eps`, 
using the unbalanced Sinkhorn algorithm [Chizat 2016] with KL-divergence terms for soft marginal constraints, with weights `(lambda1, lambda2)`
for the marginals mu, nu respectively.

A pre-computed optimal transport `plan` may be provided.

See also: [`sinkhorn_unbalanced`](@ref)
"""
function sinkhorn_unbalanced2(μ, ν, C, λ1, λ2, ε; map=nothing, plan=map, kwargs...)
    # check deprecation
    if map !== nothing
        Base.depwarn(
            "the keyword argument `map` is deprecated, please use `plan`",
            :sinkhorn_unbalanced2,
        )
    end

    γ = if plan === nothing
        sinkhorn_unbalanced(μ, ν, C, λ1, λ2, ε; kwargs...)
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
    alpha=nothing,
    beta=nothing,
    return_duals=false,
    verbose=false,
)
    if isnothing(alpha) || isnothing(beta)
        alpha = zeros(size(mu))
        beta = zeros(size(nu))
    end

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

## 1-Dimensional Optimal Transport Functions

"""
    otcost(c,mu::ContinuousUnivariateDistribution,nu::UnivariateDistribution; plan=nothing)

Compute the optimal transport cost between μ to ν, where
both are 1-Dimensional distributions and the cost
function is of the form ``c(x,y) = h(|x-y|)`` such that
``h`` is a convex function.

Return optimal transport cos
```math
\\int_0^1 c(F_\\mu^{-1}(x),F_\\nu^{-1}(x))
```

where ``F_\\mu^{-1}`` is the quantile function of ``\\mu``
and ``F_\\nu^{-1}`` is the quantile function of ``\\nu``.

A pre-computed optimal transport `plan` may be provided.
"""
function otcost(
    c, μ::ContinuousUnivariateDistribution, ν::UnivariateDistribution; plan=nothing
)
    if plan === nothing
        g(μ, ν, x) = c(quantile(μ, x), quantile(ν, x))
        f(x) = g(μ, ν, x)
        return quadgk(f, 0, 1)[1]
    else
        cost = quadgk(0, 1) do q
            x = quantile(μ, q)
            return c(x, plan(x))
        end
        return cost[1]
    end
end

"""
    otplan(c,mu::ContinuousUnivariateDistribution,nu::UnivariateDistribution)

Compute the optimal transport plan between μ to ν, where
both are 1-Dimensional distributions and the cost
function is of the form ``c(x,y) = h(|x-y|)`` such that
``h`` is a convex function.

A pre-computed optimal transport `plan` may be provided.

Return the optimal transport plan as a function
  ```math
  T(x)=F_\\nu^{-1}(F_\\mu(x))
  ```
where ``F_\\mu`` is the cdf function of ``\\mu``
and ``F_\\nu^{-1}`` is the quantile function of ``\\nu``.
"""
function otplan(c, μ::ContinuousUnivariateDistribution, ν::UnivariateDistribution)
    # Use T instead of γ to indicate that this is a Monge map.
    T(x) = quantile(ν, cdf(μ, x))
    return T
end

end