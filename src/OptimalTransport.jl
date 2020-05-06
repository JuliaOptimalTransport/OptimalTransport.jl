# OptimalTransport.jl -- optimal transportation algorithms for Julia
# Author: Stephen Zhang (syz@math.ubc.ca)

__precompile__()

module OptimalTransport

using PyCall
using Distances
using LinearAlgebra

const pot = PyNULL()

function __init__()
	copy!(pot, pyimport_conda("ot", "pot"))
end

function emd(a, b, M)
    """
    Exact solution to Kantorovich problem. Return optimal transport coupling γ
    """
    return pot.lp.emd(b, a, PyReverseDims(M))'
end

function emd2(a, b, M)
    """
    Exact solution to Kantorovich problem. Retern optimal transport cost ∫ c dα.
    """
    return pot.lp.emd2(b, a, PyReverseDims(M))
end

function sinkhorn_impl(mu, nu, C, eps; tol = 1e-6, check_marginal_step = 10, max_iter = 1000, verbose = false)
    """
    Sinkhorn algorithm to compute coupling of mu, nu with regularisation eps.
    Return dual potentials u, v such that γ_ij = u_i K_ij v_j.
    """
    K = exp.(-C/eps)
    v = ones(size(C, 2))
    u = ones(size(C, 1))
    temp_v = zeros(size(C, 1))
    temp_u = zeros(size(C, 2))
    iter = 0
    err = 0
    while true
        mul!(temp_v, K, v)
        mul!(u, Diagonal(1 ./temp_v), mu)
        mul!(temp_u, K', u)
        mul!(v, Diagonal(1 ./temp_u), nu)
        # check mu marginal
        if (iter % check_marginal_step == 0)
            err = maximum(abs.(mu - u.*(K*v)))
            if verbose
                println(string("Iteration ", iter, ", err = ", err))
            end

            if iter > max_iter
                if verbose
                    println("Warning: sinkhorn_native exited without converging")
                end
                break
            elseif err < tol
                break
            end
        end
        iter += 1
    end
    return u, v
end

function sinkhorn(mu, nu, C, eps; tol = 1e-6, check_marginal_step = 10, max_iter = 1000, verbose = false)
    """
    Sinkhorn algorithm for entropically regularised optimal transport.
    Return optimal transport coupling γ.
    """
    u, v = sinkhorn_impl(mu, nu, C, eps;
                        tol = tol, check_marginal_step = check_marginal_step, max_iter = max_iter, verbose = verbose)
    return Diagonal(u)*exp.(-C/eps)*Diagonal(v)
end

function sinkhorn2(mu, nu, C, eps; tol = 1e-6, check_marginal_step = 10, max_iter = 1000, verbose = false)
    """
    Sinkhorn algorithm for entropically regularised optimal transport.
    Return optimal transport cost ∫ c dγ + ϵ H(γ | μ ⊗ ν)
    """
    gamma = sinkhorn(mu, nu, C, eps;
                        tol = tol, check_marginal_step = check_marginal_step, max_iter = max_iter, verbose = verbose)
    return sum(gamma.*C)
end

function sinkhorn_unbalanced(μ, ν, C, λ1, λ2, ϵ; tol = 1e-6, max_iter = 1000, verbose = false)
    """
    Unbalanced Sinkhorn algorithm with KL (λ1, λ2) marginal terms for marginals
    μ, ν respectively.
    """
    @inline proxdiv_KL(s, ϵ, λ, p) = (s.^(ϵ/(ϵ + λ)) .* p.^(λ/(ϵ + λ)))./s
    a = ones(size(μ, 1))
    b = ones(size(ν, 1))
    a_old = a
    b_old = b
    tmp_a = zeros(size(ν, 1))
    tmp_b = zeros(size(μ, 1))
    K = exp.(-C/ϵ)
    iter = 1
    while true
        a_old = a
        b_old = b
        a = proxdiv_KL(mul!(tmp_b, K, b), ϵ, λ1, μ)
        b = proxdiv_KL(mul!(tmp_a, K', a), ϵ, λ2, ν)
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

function sinkhorn_unbalanced2(μ, ν, C, λ1, λ2, ϵ; tol = 1e-6, max_iter = 1000, verbose = false)
    return sum(C.*sinkhorn_unbalanced(μ, ν, C, λ1, λ2, ϵ, tol = tol, max_iter = max_iter, verbose = verbose))
end


function _sinkhorn(a, b, M, eps)
    return pot.sinkhorn(b, a, PyReverseDims(M), eps)'
end

function _sinkhorn_stabilized_epsscaling(a, b, M, eps)
    return pot.sinkhorn(b, a, PyReverseDims(M), eps, method = "sinkhorn_epsilon_scaling")'
end

function _sinkhorn_stabilized_epsscaling2(a, b, M, eps)
    return pot.sinkhorn2(b, a, PyReverseDims(M), eps, method = "sinkhorn_epsilon_scaling")[1]
end

function _sinkhorn2(a, b, M, eps)
    return pot.sinkhorn2(b, a, PyReverseDims(M), eps)[1]
end

function sinkhorn_stabilized_epsscaling(μ, ν, C, ϵ; absorb_tol = 1e3, max_iter = 10000, tol = 1e-6, λ = 0.5, k = 5, verbose = false)
    ϵ_values = [ϵ*λ^(k-j) for j = 1:k]
    α = zeros(size(μ)); β = zeros(size(ν))
    for ϵ in ϵ_values
        if verbose; println(string("Warm start: ϵ = ", ϵ)); end
        α, β = sinkhorn_stabilized(μ, ν, C, ϵ, absorb_tol = absorb_tol, max_iter = max_iter, tol = tol, α = α, β = β, return_duals = true, verbose = verbose)
    end
    K = exp.(-(C .- α .- β')/ϵ).*μ.*ν'
    return K
end

function getK(C, α, β, ϵ, μ, ν)
    return (exp.(-(C .- α .- β')/ϵ).*μ.*ν')
end

function sinkhorn_stabilized(μ, ν, C, ϵ; absorb_tol = 1e3, max_iter = 1000, tol = 1e-6, α = nothing, β = nothing, return_duals = false, verbose = false)
    if isnothing(α) || isnothing(β)
        α = zeros(size(μ)); β = zeros(size(ν))
    end

    u = ones(size(μ)); v = ones(size(ν))
    K = getK(C, α, β, ϵ, μ, ν)
    i = 0
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
