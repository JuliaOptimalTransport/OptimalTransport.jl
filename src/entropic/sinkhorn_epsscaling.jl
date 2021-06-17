struct SinkhornEpsilonScaling{A<:Sinkhorn,T<:Real} <: Sinkhorn
    alg::A
    factor::T
    steps::Int
end

function Base.show(io::IO, alg::SinkhornEpsilonScaling)
    return print(
        io, alg.alg, " with ε-scaling (steps: ", alg.steps, ", factor: ", alg.factor, ")"
    )
end

"""
    SinkhornEpsilonScaling(algorithm::Sinkhorn; factor=1//2, steps=5)

Construct an ε-scaling Sinkhorn algorithm for solving an entropically regularized optimal
transport problem.

The function uses the specified Sinkhorn `algorithm` with `steps` ε-scaling steps
with scaling factor `factor`. It sequentially solves the entropically regularized
optimal transport with regularization parameters
```math
\\varepsilon_i := \\varepsilon \\lambda^{i-k}, \\qquad (i = 1,\\ldots,k),
```
where ``\\lambda`` is the scaling factor and ``k`` the number of scaling steps.
"""
function SinkhornEpsilonScaling(alg::Sinkhorn; factor::Real=1//2, steps::Int=5)
    zero(factor) < factor < one(factor) ||
        throw(ArgumentError("the scaling `factor` must be in (0, 1)"))
    steps > 1 || throw(ArgumentError("number of scaling `steps` must be greater than 1"))
    return SinkhornEpsilonScaling(alg, factor, steps)
end

function sinkhorn(μ, ν, C, ε, alg::SinkhornEpsilonScaling; kwargs...)
    # unpack
    scaling_factor = alg.factor
    scaling_steps = alg.steps

    # initial regularization parameter
    εstep = ε * scaling_factor^(1 - scaling_steps)
    @debug "ε-scaling Sinkhorn algorithm: ε = $εstep"

    # initialize solver and perform Sinkhorn algorithm
    solver = build_solver(μ, ν, C, εstep, alg.alg; kwargs...)
    solve!(solver)

    for _ in 2:(scaling_steps - 1)
        εstep *= scaling_factor
        @debug "ε-scaling Sinkhorn algorithm: ε = $εstep"

        # re-run Sinkhorn algorithm with smaller regularization parameter
        solver = update_epsilon!(solver, εstep)
        solve!(solver)
    end

    # re-run Sinkhorn algorithm with desired regularization parameter
    @debug "ε-scaling Sinkhorn algorithm: ε = $ε"
    solver = update_epsilon!(solver, oftype(εstep, ε))
    solve!(solver)

    # compute final plan
    γ = plan(solver)

    return γ
end

# support for `SinkhornGibbs` and `SinkhornStabilized`
function update_epsilon!(solver::SinkhornSolver{SinkhornGibbs}, ε::Real)
    cache = solver.cache
    @. cache.K = exp(-solver.C / ε)
    newsolver = SinkhornSolver(
        solver.source,
        solver.target,
        solver.C,
        ε,
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

function update_epsilon!(solver::SinkhornSolver{<:SinkhornStabilized}, ε::Real)
    newsolver = SinkhornSolver(
        solver.source,
        solver.target,
        solver.C,
        ε,
        solver.alg,
        solver.atol,
        solver.rtol,
        solver.maxiter,
        solver.check_convergence,
        solver.cache,
        solver.convergence_cache,
    )
    absorb!(newsolver)
    return newsolver
end

# deprecations

"""
    sinkhorn_stabilized_epsscaling(
        μ, ν, C, ε;
        scaling_factor=1//2, scaling_steps=5, absorb_tol=1_000, kwargs...
    )

This method is deprecated, please use
```julia
sinkhorn(
    SinkhornEpsilonScaling(
        SinkhornStabilized(; absorb_tol=absorb_tol);
        factor=scaling_factor,
        steps=scaling_steps,
    ),
    μ,
    ν,
    C,
    ε;
    kwargs...,
)
```
instead.

See also: [`sinkhorn`](@ref), [`SinkhornEpsilonScaling`](@ref)
"""
function sinkhorn_stabilized_epsscaling(
    μ,
    ν,
    C,
    ε;
    lambda=nothing,
    scaling_factor=lambda,
    k=nothing,
    scaling_steps=k,
    absorb_tol=1_000,
    kwargs...,
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

    Base.depwarn(
        "`sinkhorn_stabilized_epsscaling` is deprecated, " *
        "please use `sinkhorn` with `SinkhornEpsilonScaling` instead",
        :sinkhorn_stabilized_epsscaling,
    )

    # construct ε-scaling algorithm
    factor = scaling_factor === nothing ? 1//2 : scaling_factor
    steps = scaling_steps === nothing ? 5 : scaling_steps
    algorithm = SinkhornEpsilonScaling(
        SinkhornStabilized(; absorb_tol=absorb_tol); factor=factor, steps=steps
    )

    return sinkhorn(μ, ν, C, ε, algorithm; kwargs...)
end
