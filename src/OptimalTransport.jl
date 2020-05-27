# OptimalTransport.jl -- optimal transportation algorithms for Julia
# Author: Stephen Zhang (syz@math.ubc.ca)
module OptimalTransport

using PyCall
using Distances
using LinearAlgebra

const pot = PyNULL()

function __init__()
	copy!(pot, pyimport_conda("ot", "pot", "conda-forge"))
end

export emd

"""
    emd(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64})

*Wrapper to POT function* Exact solution to Kantorovich problem with marginals `a` and `b` and a cost matrix `M` of dimensions
`(length(a), length(b))`.

Return optimal transport coupling `γ` of the same dimensions as `M`.
"""
function emd(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64})
    return pot.lp.emd(b, a, PyReverseDims(M))'
end

export emd2

"""
    emd2(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64})

*Wrapper to POT function* Same as `emd`, but returns instead the cost of the optimal transport, i.e. `sum(M.*γ)`.
"""
function emd2(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64})
    return pot.lp.emd2(b, a, PyReverseDims(M))
end

export sinkhorn_impl

"""
    sinkhorn_impl(mu::Vector{Float64}, nu::Vector{Float64}, K::Matrix{Float64}, u::Vector{Float64}, v::Vector{Float64}, eps::Float64; tol = 1e-6, check_marginal_step = 10, max_iter = 1000, verbose = false)

Sinkhorn algorithm to compute coupling of `mu`, `nu` with entropic regularisation parameter `eps`.
`u` and `v` are arrays to be filled with the values for the dual potentials for `mu` and `nu` respectively.

Return dual potentials `u`, `v` such that `γ = Diagonal(u)*K*Diagonal(v)`, where `K = exp.(-C/eps)` is the Gibbs kernel.
"""
function sinkhorn_gibbs(mu, nu, K; tol=1e-9, check_marginal_step=10, maxiter=1000)
    if !(sum(mu) ≈ sum(nu))
        throw(ArgumentError("Error: mu and nu must lie in the simplex"))
    end

    # initial iteration
    temp_v = vec(sum(K; dims = 2))
    u = mu ./ temp_v
    temp_u = vec(sum(K; dims = 1))
    v = nu ./ temp_u

    isconverged = false
    for iter in 0:maxiter
        # check mu marginal
        if iter % check_marginal_step == 0
            mul!(temp_v, K, v)
            @. temp_v = abs(mu - u * temp_v)

            err = maximum(temp_v)
            @debug "Sinkhorn algoritm: iteration $iter" err

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
        @debug "Warning: Sinkhorn algorithm did not converge"
    end

    return u, v
end

"""
    sinkhorn(mu::Vector{Float64}, nu::Vector{Float64}, C::Matrix{Float64}, eps::Float64; tol = 1e-6, check_marginal_step = 10, max_iter = 1000, verbose = false)

Sinkhorn algorithm to compute coupling of `mu`, `nu` with entropic regularisation parameter `eps`.
Return optimal transport coupling `γ`.
"""
function sinkhorn(mu, nu, C, eps; kwargs...)
    # compute Gibbs kernel
    K = @. exp(-C / eps)

    # compute dual potentials
    u, v = sinkhorn_gibbs(mu, nu, K; kwargs...)

    return Diagonal(u) * K * Diagonal(v)
end

"""
    sinkhorn2(mu::Vector{Float64}, nu::Vector{Float64}, C::Matrix{Float64}, eps::Float64; tol = 1e-6, check_marginal_step = 10, max_iter = 1000, verbose = false)

Sinkhorn algorithm to compute coupling of `mu`, `nu` with entropic regularisation parameter `eps`.
Return optimal transport cost.
"""
function sinkhorn2(mu, nu, C, eps; kwargs...)
    gamma = sinkhorn(mu, nu, C, eps; kwargs...)
    return dot(gamma, C)
end

export sinkhorn_unbalanced

"""
    sinkhorn_unbalanced(μ::Vector{Float64}, ν::Vector{Float64}, C::Matrix{Float64}, λ1::Float64, λ2::Float64, ϵ::Float64; tol = 1e-6, max_iter = 1000, verbose = false)

Unbalanced Sinkhorn algorithm with KL-divergence terms for soft marginal constraints, with weights `(λ1, λ2)`
for μ, ν respectively.

Returns the optimal transport plan.
"""
function sinkhorn_unbalanced(μ::Vector{Float64}, ν::Vector{Float64}, C::Matrix{Float64}, λ1::Float64, λ2::Float64, ϵ::Float64; tol = 1e-6, max_iter = 1000, verbose = false)
    function proxdiv_KL(out, s, ϵ, λ, p)
        for i = 1:size(s, 1)
            out[i] = (s[i]^(ϵ/(ϵ + λ)) * p[i]^(λ/(ϵ + λ)))/s[i]
        end
    end

    a = ones(size(μ, 1))
    b = ones(size(ν, 1))
    a_old = a
    b_old = b
    tmp_a = zeros(size(ν, 1))
    tmp_b = zeros(size(μ, 1))

    K = zeros(size(C));
    for i = 1:size(C, 1), j = 1:size(C, 2)
        K[i, j] = exp(-C[i, j]/ϵ)
    end

    iter = 1

    if !(sum(μ) ≈ sum(ν))
        throw(ArgumentError("Error: μ and ν must lie in the simplex"))
    end

    while true
        a_old = a
        b_old = b
        mul!(tmp_b, K, b)
        proxdiv_KL(a, tmp_b, ϵ, λ1, μ)
        mul!(tmp_a, K', a)
        proxdiv_KL(b, tmp_a, ϵ, λ2, ν)
        iter += 1
        if iter % 10 == 0
            err_a = maximum(abs.(a - a_old))/max(maximum(abs.(a)), maximum(abs.(a_old)), 1)
            err_b = maximum(abs.(b - b_old))/max(maximum(abs.(b)), maximum(abs.(b_old)), 1)
            if verbose
                println(string("Iteration ", iter, ", err = ", 0.5*(err_a + err_b)))
            end
            if (0.5*(err_a + err_b) < tol) || iter > max_iter
                break
            end
        end
    end
    if iter > max_iter && verbose
        println("Warning: exited before convergence")
    end
    return Diagonal(a)*K*Diagonal(b)
end

export sinkhorn_unbalanced2

"""
    sinkhorn_unbalanced2(μ::Vector{Float64}, ν::Vector{Float64}, C::Matrix{Float64}, λ1::Float64, λ2::Float64, ϵ::Float64; tol = 1e-6, max_iter = 1000, verbose = false)

Same as `sinkhorn_unbalanced`, except return the corresponding cost.
"""
function sinkhorn_unbalanced2(μ::Vector{Float64}, ν::Vector{Float64}, C::Matrix{Float64}, λ1::Float64, λ2::Float64, ϵ::Float64; tol = 1e-6, max_iter = 1000, verbose = false)
    return sum(C.*sinkhorn_unbalanced(μ, ν, C, λ1, λ2, ϵ, tol = tol, max_iter = max_iter, verbose = verbose))
end

export _sinkhorn

"""
    _sinkhorn(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64}, eps::Float64)

Wrapper to POT function `sinkhorn`
"""
function _sinkhorn(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64}, eps::Float64)
    return pot.sinkhorn(b, a, PyReverseDims(M), eps)'
end

export _sinkhorn2

"""
    _sinkhorn2(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64}, eps::Float64)

Wrapper to POT function `sinkhorn2`
"""
function _sinkhorn2(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64}, eps::Float64)
    return pot.sinkhorn2(b, a, PyReverseDims(M), eps)[1]
end

export _sinkhorn_unbalanced

"""
    _sinkhorn_unbalanced(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64}, reg::Float64, reg_m::Float64)

Wrapper to POT function `sinkhorn_unbalanced`
"""
function _sinkhorn_unbalanced(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64}, reg::Float64, reg_m::Float64)
    return pot.sinkhorn_unbalanced(b, a, PyReverseDims(M), reg, reg_m)'
end

export _sinkhorn_unbalanced2

"""
    _sinkhorn_unbalanced2(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64}, reg::Float64, reg_m::Float64)

Wrapper to POT function `sinkhorn_unbalanced2`
"""
function _sinkhorn_unbalanced2(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64}, reg::Float64, reg_m::Float64)
    return pot.sinkhorn_unbalanced2(b, a, PyReverseDims(M), reg, reg_m)
end

export _sinkhorn_stabilized_epsscaling

"""
    _sinkhorn_stabilized_epsscaling(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64}, eps::Float64)

Wrapper to POT function `sinkhorn` with method set to `sinkhorn_epsilon_scaling`
"""
function _sinkhorn_stabilized_epsscaling(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64}, eps::Float64)
    return pot.sinkhorn(b, a, PyReverseDims(M), eps, method = "sinkhorn_epsilon_scaling")'
end

export _sinkhorn_stabilized_epsscaling2

"""
    _sinkhorn_stabilized_epsscaling2(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64}, eps::Float64)

Wrapper to POT function `sinkhorn2` with method set to `sinkhorn_epsilon_scaling`
"""
function _sinkhorn_stabilized_epsscaling2(a::Vector{Float64}, b::Vector{Float64}, M::Matrix{Float64}, eps::Float64)
    return pot.sinkhorn2(b, a, PyReverseDims(M), eps, method = "sinkhorn_epsilon_scaling")[1]
end

export sinkhorn_stabilized_epsscaling

"""
    sinkhorn_stabilized_epsscaling(μ::Vector{Float64}, ν::Vector{Float64}, C::Matrix{Float64}, ϵ::Float64; absorb_tol = 1e3, max_iter = 10000, tol = 1e-6, λ = 0.5, k = 5, verbose = false)

Stabilized Sinkhorn algorithm with epsilon-scaling.
"""
function sinkhorn_stabilized_epsscaling(μ::Vector{Float64}, ν::Vector{Float64}, C::Matrix{Float64}, ϵ::Float64; absorb_tol = 1e3, max_iter = 10000, tol = 1e-6, λ = 0.5, k = 5, verbose = false)
    ϵ_values = [ϵ*λ^(k-j) for j = 1:k]
    α = zeros(size(μ)); β = zeros(size(ν))
    for ϵ in ϵ_values
        if verbose; println(string("Warm start: ϵ = ", ϵ)); end
        α, β = sinkhorn_stabilized(μ, ν, C, ϵ, absorb_tol = absorb_tol, max_iter = max_iter, tol = tol, α = α, β = β, return_duals = true, verbose = verbose)
    end
    K = exp.(-(C .- α .- β')/ϵ).*μ.*ν'
    return K
end

function getK(C::Matrix{Float64}, α::Vector{Float64}, β::Vector{Float64}, ϵ::Float64, μ::Vector{Float64}, ν::Vector{Float64})
    return (exp.(-(C .- α .- β')/ϵ).*μ.*ν')
end

export sinkhorn_stabilized

"""
    sinkhorn_stabilized(μ::Vector{Float64}, ν::Vector{Float64}, C::Matrix{Float64}, ϵ::Float64; absorb_tol = 1e3, max_iter = 1000, tol = 1e-6, α = nothing, β = nothing, return_duals = false, verbose = false)

Stabilized Sinkhorn algorithm.
"""
function sinkhorn_stabilized(μ::Vector{Float64}, ν::Vector{Float64}, C::Matrix{Float64}, ϵ::Float64; absorb_tol = 1e3, max_iter = 1000, tol = 1e-6, α = nothing, β = nothing, return_duals = false, verbose = false)
    if isnothing(α) || isnothing(β)
        α = zeros(size(μ)); β = zeros(size(ν))
    end

    u = ones(size(μ)); v = ones(size(ν))
    K = getK(C, α, β, ϵ, μ, ν)
    i = 0

    if !(sum(μ) ≈ sum(ν))
        throw(ArgumentError("Error: μ and ν must lie in the simplex"))
    end

    while true
        u = μ./(K*v .+ 1e-16)
        v = ν./(K'*u .+ 1e-16)
        if (max(norm(u, Inf), norm(v, Inf)) > absorb_tol)
            if verbose; println("Absorbing (u, v) into (α, β)"); end
            # absorb into α, β
            α = α + ϵ*log.(u); β = β + ϵ*log.(v)
            u = ones(size(μ)); v = ones(size(ν))
            K = getK(C, α, β, ϵ, μ, ν)
        end
        if i % 10 == 0
            # check marginal
            γ = getK(C, α, β, ϵ, μ, ν).*(u.*v')
            err_μ = norm(γ*ones(size(ν)) - μ, Inf)
            err_ν = norm(γ'*ones(size(μ)) - ν, Inf)
            if verbose; println(string("Iteration ", i, ", err = ", 0.5*(err_μ + err_ν))); end
            if 0.5*(err_μ + err_ν) < tol
                break
            end
        end

        if i > max_iter
            if verbose; println("Warning: exited before convergence"); end
            break
        end
        i+=1
    end
    α = α + ϵ*log.(u); β = β + ϵ*log.(v)
    if return_duals
        return α, β
    end
    return getK(C, α, β, ϵ, μ, ν)
end

end
