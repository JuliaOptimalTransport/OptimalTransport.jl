"""
    proxdivKL!(s, p, ε, λ)

Operator ``\\operatorname{proxdiv}_F(s, p, ε)`` associated with the marginal penalty
``q \\mapsto \\lambda \\operatorname{KL}(q | p)``. For further details see [^CPSV18]. 

[^CPSV18]: Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F.-X. (2018). [Scaling algorithms for unbalanced optimal transport problems](https://doi.org/10.1090/mcom/3303). Mathematics of Computation, 87(314), 2563–2609.
"""
proxdivKL!(s, p, ε, λ) = (s .= (p ./ s) .^ (λ / (ε + λ)))

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
    proxdivF1!(s, p, ε) = proxdivKL!(s, p, ε, λ1)
    proxdivF2!(s, p, ε) = proxdivKL!(s, p, ε, λ2)

    return sinkhorn_unbalanced(μ, ν, C, proxdivF1!, proxdivF2!, ε; kwargs...)
end

function sinkhorn_unbalanced(μ, C, λ::Real, ε; kwargs...)
    proxdivF!(s, p, ε) = proxdivKL!(s, p, ε, λ)
    return sinkhorn_unbalanced(μ, C, proxdivF!, ε; kwargs...)
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
    function sinkhorn_unbalanced(
            μ, C, proxdivF!, ε; atol = nothing, rtol = nothing, maxiter::Int = 1_000, check_convergence::Int=10
    )

    Specialised case of [`sinkhorn_unbalanced`](@ref) to the special symmetric case where both inputs `μ, ν` are identical and the cost `C` is symmetric.
    This implementation takes advantage of additional structure in the symmetric case which allows for a fixed point iteration with much faster convergence,
    similar to that described by [^FeydyP19] and also employed in [`sinkhorn_divergence`](@ref) for the balanced case.

    [^FeydyP19]: Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé, and Gabriel Peyré. Interpolating between optimal transport and mmd using sinkhorn divergences. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 2681–2690. PMLR, 2019.
"""
function sinkhorn_unbalanced(
    μ,
    C,
    proxdivF!,
    ε;
    atol=nothing,
    rtol=nothing,
    maxiter::Int=1_000,
    check_convergence::Int=10,
)
    # compute Gibbs kernel
    K = @. exp(-C / ε)

    # set default values of squared tolerances
    T = float(Base.promote_eltype(μ, K))
    sqatol = atol === nothing ? 0 : atol^2
    sqrtol = rtol === nothing ? (sqatol > zero(sqatol) ? zero(T) : eps(T)) : rtol^2

    # initialize iterate and cache
    a = similar(μ, T)
    sum!(a, K)
    tmp = similar(a)

    isconverged = false
    for iter in 1:maxiter
        ischeck = iter % check_convergence == 0
        mul!(tmp, K, a)
        proxdivF!(tmp, μ, ε)
        if ischeck
            sqnorm_a = sum(abs2, tmp)
            sqnorm_a_old = sum(abs2, a)
            sqeuclidean_a = sum(abs2, a - tmp)
            @debug "Sinkhorn algorithm (" *
                string(iter) *
                "/" *
                string(maxiter) *
                ": squared Euclidean distance of iterates = " *
                string(sqeuclidean_a)

            # check convergence of `a`
            if sqeuclidean_a < max(sqatol, sqrtol * max(sqnorm_a, sqnorm_a_old))
                @debug "Sinkhorn algorithm ($iter/$maxiter): converged"
                isconverged = true
                break
            end
        end
        @. a = exp.(0.5 * (log.(a) + log.(tmp)))
    end
    if !isconverged
        @warn "Sinkhorn algorithm ($maxiter/$maxiter): not converged"
    end
    return K .* a .* a'
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
    μ, ν, c, λ1_or_proxdivf1, λ2_or_proxdivf2, ε; plan=nothing, kwargs...
)
    γ = if plan === nothing
        sinkhorn_unbalanced(μ, ν, c, λ1_or_proxdivf1, λ2_or_proxdivf2, ε; kwargs...)
    else
        # check dimensions
        size(c) == (length(μ), length(ν)) ||
            error("cost matrix `c` must be of size `(length(μ), length(ν))`")
        size(plan) == size(c) || error(
            "optimal transport plan `plan` and cost matrix `c` must be of the same size",
        )
        plan
    end
    return dot(γ, c)
end

function sinkhorn_unbalanced2(μ, c, λ_or_proxdivf, ε; plan=nothing, kwargs...)
    γ = if plan === nothing
        sinkhorn_unbalanced(μ, c, λ_or_proxdivf, ε; kwargs...)
    else
        # check dimensions
        size(c) == (length(μ), length(μ)) ||
            error("cost matrix `c` must be of size `(length(μ), length(μ))`")
        size(plan) == size(c) || error(
            "optimal transport plan `plan` and cost matrix `c` must be of the same size",
        )
        plan
    end
    return dot(γ, c)
end

"""
    sinkhorn_divergence_unbalanced(μ, ν, cμν, cμ, cν, λ, ε; kwargs...)

Compute the unbalanced Sinkhorn divergence between unnormalized inputs `μ` and `ν` with cost matrix `cμν`, `cμ` and `cν` between `(μ,ν)`, `(μ, μ)` and `(ν, ν)` respectively,
regularization level `ε` and marginal constraint parameter `λ`. Following [^SFVTP19], the unbalanced Sinkhorn divergence is defined as
```math
    \\operatorname{S}_{\\varepsilon, \\lambda} (\\mu, \\nu) := \\operatorname{OT}_{ε, λ}(μ,ν)
    - \\frac{1}{2}(\\operatorname{OT}_{ε, λ}(μ,μ) + \\operatorname{OT}_{ε, λ}(ν,ν)) + \\frac{ε}{2}(m(μ) + m(ν))^2,
```
where ``\\operatorname{OT}_{ε, λ}(\\alpha, \\beta)`` is defined to be 
```math
        \\operatorname{OT}_{ε, λ}(\\alpha, \\beta) = \\inf_{\\gamma} \\langle C, \\gamma \\rangle + \\varepsilon \\operatorname{KL}(\\gamma | \\alpha \\otimes \\beta) + \\lambda ( \\operatorname{KL}(\\gamma_1 | \\alpha) + \\operatorname{KL}(\\gamma_2 | \\beta) ),
```
i.e. the output of calling `sinkhorn_unbalanced2` with the default Kullback-Leibler marginal penalties. 

[^SFVTP19]: Séjourné, T., Feydy, J., Vialard, F.X., Trouvé, A. and Peyré, G., 2019. Sinkhorn divergences for unbalanced optimal transport. arXiv preprint arXiv:1910.12958.
"""
function sinkhorn_divergence_unbalanced(μ, ν, cμν, cμ, cν, λ, ε; kwargs...)
    Sμν = sinkhorn_unbalanced2(μ, ν, cμν, λ, λ, ε; kwargs...)
    Sμ = sinkhorn_unbalanced2(μ, cμ, λ, ε; kwargs...)
    Sν = sinkhorn_unbalanced2(ν, cν, λ, ε; kwargs...)
    return max(0, Sμν - (Sμ + Sν) / 2 + ε * (sum(μ) - sum(ν))^2 / 2)
end
