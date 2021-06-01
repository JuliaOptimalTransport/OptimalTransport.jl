"""
    ot_plan(c, μ, ν; kwargs...)

Compute the optimal transport plan for the Monge-Kantorovich problem with source and target
marginals `μ` and `ν` and cost `c`.

The optimal transport plan solves
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\int c(x, y) \\, \\mathrm{d}\\gamma(x, y)
```
where ``\\Pi(\\mu, \\nu)`` denotes the couplings of ``\\mu`` and ``\\nu``.

See also: [`ot_cost`](@ref)
"""
ot_plan

"""
    ot_cost(c, μ, ν; kwargs...)

Compute the optimal transport cost for the Monge-Kantorovich problem with source and target
marginals `μ` and `ν` and cost `c`.

The optimal transport cost is the scalar value
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\int c(x, y) \\, \\mathrm{d}\\gamma(x, y)
```
where ``\\Pi(\\mu, \\nu)`` denotes the couplings of ``\\mu`` and ``\\nu``.

See also: [`ot_plan`](@ref)
"""
ot_cost

#############
# Discrete OT
#############

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

###################################
# Semidiscrete and continuous 1D OT
###################################

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

################
# Discrete 1D OT
################

# internal iterator for discrete one-dimensional OT problems
# it returns tuples that consist of the indices of the source and target histograms
# and the optimal flow between the corresponding points
struct Discrete1DOTIterator{T,M,N}
    mu::M
    nu::N
end

# histograms `μ` and `ν` are expected to be iterators of the histograms where the
# corresponding support is sorted
function Discrete1DOTIterator(μ, ν)
    T = Base.promote_eltype(μ, ν)
    return Discrete1DOTIterator{T,typeof(μ),typeof(ν)}(μ, ν)
end

Base.IteratorEltype(::Type{<:Discrete1DOTIterator}) = Base.HasEltype()
Base.eltype(::Type{<:Discrete1DOTIterator{T}}) where {T} = Tuple{Int,Int,T}

Base.length(d::Discrete1DOTIterator) = length(d.mu) + length(d.nu) - 1

# we iterate through the source and target histograms
function Base.iterate(
    d::Discrete1DOTIterator{T}, (i, j, μnext, νnext)=(1, 1, iterate(d.mu), iterate(d.nu))
) where {T}
    # if we are done with iterating through the source and/or target histogram,
    # iteration is stopped
    if μnext === nothing || νnext === nothing
        return nothing
    end

    # unpack next values and states of the source and target histograms
    μiter, μstate = μnext
    νiter, νstate = νnext

    # compute next value of the iterator: indices of source and target histograms
    # and optimal flow between the corresponding points
    min_iter, max_iter = minmax(μiter, νiter)
    iter = (i, j, min_iter)

    # compute next state of the iterator
    diff = max_iter - min_iter
    state = if μiter < max_iter
        # move forward in the source histogram
        (i + 1, j, iterate(d.mu, μstate), (diff, νstate))
    else
        # move forward in the target histogram
        (i, j + 1, (diff, μstate), iterate(d.nu, νstate))
    end

    return iter, state
end

"""
    ot_plan(c, μ::DiscreteNonParametric, ν::DiscreteNonParametric)

Compute the optimal transport cost for the Monge-Kantorovich problem with univariate
discrete distributions `μ` and `ν` as source and target marginals and cost function `c`
of the form ``c(x, y) = h(|x - y|)`` where ``h`` is a convex function.

In this setting, the optimal transport plan can be computed analytically. It is returned as
a sparse matrix.

See also: [`ot_cost`](@ref), [`emd`](@ref)
"""
function ot_plan(_, μ::DiscreteNonParametric, ν::DiscreteNonParametric)
    # unpack the probabilities of the two distributions
    μprobs = probs(μ)
    νprobs = probs(ν)

    # create the iterator
    # note: support of `DiscreteNonParametric` is sorted
    iter = Discrete1DOTIterator(μprobs, νprobs)

    # create arrays for the indices of the two histograms and the optimal flow between the
    # corresponding points
    n = length(iter)
    I = Vector{Int}(undef, n)
    J = Vector{Int}(undef, n)
    W = Vector{Base.promote_eltype(μprobs, νprobs)}(undef, n)

    # compute the sparse optimal transport plan
    @inbounds for (idx, (i, j, w)) in enumerate(iter)
        I[idx] = i
        J[idx] = j
        W[idx] = w
    end
    γ = sparse(I, J, W, length(μprobs), length(νprobs))

    return γ
end

"""
    ot_cost(
        c, μ::DiscreteNonParametric, ν::DiscreteNonParametric; plan=nothing
    )

Compute the optimal transport cost for the Monge-Kantorovich problem with discrete
univariate distributions `μ` and `ν` as source and target marginals and cost function `c`
of the form ``c(x, y) = h(|x - y|)`` where ``h`` is a convex function.

In this setting, the optimal transport cost can be computed analytically.

A pre-computed optimal transport `plan` may be provided.

See also: [`ot_plan`](@ref), [`emd2`](@ref)
"""
function ot_cost(c, μ::DiscreteNonParametric, ν::DiscreteNonParametric; plan=nothing)
    return _ot_cost(c, μ, ν, plan)
end

# compute cost from scratch if no plan is provided
function _ot_cost(c, μ::DiscreteNonParametric, ν::DiscreteNonParametric, ::Nothing)
    # unpack the probabilities of the two distributions
    μprobs = probs(μ)
    νprobs = probs(ν)

    # create the iterator
    # note: support of `DiscreteNonParametric` is sorted
    iter = Discrete1DOTIterator(μprobs, νprobs)

    # compute the cost
    μsupport = support(μ)
    νsupport = support(ν)
    cost = sum(w * c(μsupport[i], νsupport[j]) for (i, j, w) in iter)

    return cost
end

# if a sparse plan is provided, we just iterate through the non-zero entries
function _ot_cost(
    c, μ::DiscreteNonParametric, ν::DiscreteNonParametric, plan::SparseMatrixCSC
)
    # extract non-zero flows
    I, J, W = findnz(plan)

    # compute the cost
    μsupport = support(μ)
    νsupport = support(ν)
    cost = sum(w * c(μsupport[i], νsupport[j]) for (i, j, w) in zip(I, J, W))

    return cost
end

# fallback: compute cost matrix (probably often faster to compute cost from scratch)
function _ot_cost(c, μ::DiscreteNonParametric, ν::DiscreteNonParametric, plan)
    return dot(plan, StatsBase.pairwise(c, support(μ), support(ν)))
end
