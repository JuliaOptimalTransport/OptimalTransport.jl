struct SinkhornDual <: Sinkhorn
    stabilized::Bool
    show_trace::Bool
end

Base.show(io::IO, ::SinkhornDual) = print(io, "Dual ascent entropy regularisation")

struct SinkhornDualCache{XT,KT}
    x::XT
    K::KT
end

function build_cache(
    ::Type{T},
    alg::SinkhornDual,
    size2::Tuple,
    μ::AbstractVecOrMat,
    ν::AbstractVecOrMat,
    C::AbstractMatrix,
    ε::Real,
) where {T}
    # compute Gibbs kernel 
    K = similar(C, T)
    if alg.stabilized
        @. K = -C / ε
    else
        @. K = exp(-C / ε)
    end
    # create and initialize dual potentials
    x = similar(μ, T, size(μ, 1) + size(ν, 1))
    fill!(x, zero(T))
    return SinkhornDualCache(x, K)
end

function solve!(solver::SinkhornSolver{<:SinkhornDual})
    # unpack solver
    μ = solver.source
    ν = solver.target
    eps = solver.eps
    cache = solver.cache
    atol = solver.atol
    rtol = solver.rtol
    maxiter = solver.maxiter
    check_convergence = solver.check_convergence

    # unpack cache
    x = cache.x
    K = cache.K
    # dual problem <μ, u> + <ν, v> - OT*(u, v)
    function F(x, μ, ν, K, eps)
        return dot(μ, x[1:size(μ, 1)]) +
               dot(ν, x[(size(μ, 1) + 1):end]) +
               Dual.ot_entropic_dual(
                   -x[1:size(μ, 1)],
                   -x[(size(μ, 1) + 1):end],
                   eps,
                   K;
                   stabilized=solver.alg.stabilized,
               )
    end
    opt = optimize(
        x -> F(x, μ, ν, K, eps),
        x,
        Optim.LBFGS(),
        Optim.Options(;
            x_tol=atol, f_tol=rtol, iterations=maxiter, show_trace=solver.alg.show_trace
        );
        autodiff=:forward,
    )
    x .= Optim.minimizer(opt)
    return nothing
end

function plan(solver::SinkhornSolver{SinkhornDual})
    cache = solver.cache
    u = cache.x[1:size(solver.source, 1)]
    v = cache.x[(size(solver.source, 1) + 1):end]
    ε = solver.eps
    if solver.alg.stabilized
        γ = LogExpFunctions.softmax(
            cache.K .+ add_singleton(-u / ε, Val(2)) .+ add_singleton(-v / ε, Val(1))
        )
    else
        γ =
            cache.K .* add_singleton(exp.(-u / ε), Val(2)) .*
            add_singleton(exp.(-v / ε), Val(1))
    end
    return γ / sum(γ)
end
