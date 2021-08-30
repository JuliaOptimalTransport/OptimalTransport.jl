struct SinkhornDual <: Sinkhorn end

Base.show(io::IO, ::SinkhornDual) = print(io, "Dual ascent entropy regularisation")

function ot_entropic_dual(u, v, K, ε)
    # (μ, ν) → min_{γ ∈ Π(μ, ν)} ε H(γ | K)
    # has Legendre transform
    # (u, v) → ε < exp(u/ε), K exp(v/ε) >
    return ε * dot(exp.(u/ε), K * exp.(v/ε))
end

function ot_entropic_semidual(μ, v, K, ε) 
    # ν → min_{γ ∈ Π(μ, ν)} ε H(γ | K)
    # has Legendre transform
    # v → -ε <μ, log(α/(Ke^{u/ε})) - 1>
    return -ε * dot(μ, log( μ ./ (K * e^(u/ε))) - 1) 
end

struct SinkhornDualCache{XT,KT}
    x::XT
    K::KT
end

function build_cache(
    ::Type{T},
    ::SinkhornDual,
    size2::Tuple,
    μ::AbstractVecOrMat,
    ν::AbstractVecOrMat,
    C::AbstractMatrix,
    ε::Real,
) where {T}
    # compute Gibbs kernel 
    K = similar(C, T)
    @. K = exp(-C / ε)

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
    F(x, μ, ν, K, eps) = dot(μ, x[1:size(μ, 1)]) + dot(ν, x[size(μ, 1)+1:end]) + ot_entropic_dual(-x[1:size(μ, 1)], -x[size(μ, 1)+1:end], K, eps)
    opt = optimize( x -> F(x, μ, ν, K, eps), 
              x,
              Optim.LBFGS(),
              Optim.Options(; iterations = maxiter);
              autodiff=:forward
             )
    x .= Optim.minimizer(opt)    
    return nothing
end

function plan(solver::SinkhornSolver{SinkhornDual})
    cache = solver.cache
    u = cache.x[1:size(solver.source, 1)]
    v = cache.x[size(solver.source, 1)+1:end]
    ε = solver.eps
    γ = cache.K .* add_singleton(exp.(-u/ε), Val(2)) .* add_singleton(exp.(-v/ε), Val(1))
    return γ
end
