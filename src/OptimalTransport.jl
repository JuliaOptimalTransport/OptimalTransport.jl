# OptimalTransport.jl -- optimal transportation algorithms for Julia
# Author: Stephen Zhang (syz@math.ubc.ca)
# Contributors: David Widmann (@devmotion) Tetsuya Matsumoto (tmatsumoto1997@gmail.com)

module OptimalTransport

using PyCall
using Distances
using LinearAlgebra
using IterativeSolvers
using SparseArrays

export sinkhorn, sinkhorn2, pot_sinkhorn, pot_sinkhorn2
export emd, emd2
export sinkhorn_stabilized, sinkhorn_stabilized_epsscaling
export sinkhorn_unbalanced, sinkhorn_unbalanced2, pot_sinkhorn_unbalanced, pot_sinkhorn_unbalanced2

const pot = PyNULL()

function __init__()
	copy!(pot, pyimport_conda("ot", "pot", "conda-forge"))
end


"""
    emd(mu, nu, C)

Compute exact transport map for Kantorovich problem with marginals `mu` and `nu` and a cost matrix `C` of dimensions
`(length(mu), length(nu))`.

Return optimal transport coupling of the same dimensions as `C`.

This function is a wrapper of the function
[`emd`](https://pythonot.github.io/all.html#ot.emd) in the Python Optimal Tranpsort package.
"""
function emd(mu, nu, C)
    return pot.lp.emd(nu, mu, PyReverseDims(C))'
end


"""
    emd2(mu, nu, C)

Compute exact transport cost for Kantorovich problem with marginals `mu` and `nu` and a cost matrix `C` of dimensions
`(length(mu), length(nu))`.

This function is a wrapper of the function
[`emd2`](https://pythonot.github.io/all.html#ot.emd2) in the Python Optimal Transport package.
"""
function emd2(mu, nu, C)
    return pot.lp.emd2(nu, mu, PyReverseDims(C))[1]
end

"""
    sinkhorn_gibbs(mu, nu, K; kwargs...)

Compute dual potentials `u` and `v` for histograms `mu` and `nu` and Gibbs kernel `K` using
the Sinkhorn algorithm.

The Gibbs kernel `K` is given by `K = exp.(- C / eps)` where `C` is the cost matrix and
`eps` the entropic regularization parameter. The optimal transport map for histograms `u`
and `v` and cost matrix `C` with regularization parameter `eps` can be computed as
`Diagonal(u) * K * Diagonal(v)`.
"""
function sinkhorn_gibbs(mu, nu, K; tol=1e-9, check_marginal_step=10, maxiter=1000)
    if !(sum(mu) ≈ sum(nu))
        throw(ArgumentError("Error: mu and nu must lie in the simplex"))
    end

    # initial iteration
    temp_v = vec(sum(K; dims = 2))
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
    sinkhorn(mu, nu, C, eps; kwargs...)

Compute optimal transport map of histograms `mu` and `nu` with cost matrix `C` and entropic
regularization parameter `eps`.
"""
function sinkhorn(mu, nu, C, eps; kwargs...)
    # compute Gibbs kernel
    K = @. exp(-C / eps)

    # compute dual potentials
    u, v = sinkhorn_gibbs(mu, nu, K; kwargs...)

    return Diagonal(u) * K * Diagonal(v)
end

"""
    sinkhorn2(mu, nu, C, eps; kwargs...)

Compute optimal transport cost of histograms `mu` and `nu` with cost matrix `C` and entropic
regularization parameter `eps`.
"""
function sinkhorn2(mu, nu, C, eps; kwargs...)
    gamma = sinkhorn(mu, nu, C, eps; kwargs...)
    return dot(gamma, C)
end


"""
    sinkhorn_unbalanced(mu, nu, C, lambda1, lambda2, eps; kwargs...)

Computes the optimal transport map of histograms `mu` and `nu` with cost matrix `C` and entropic regularization parameter `eps`, 
using the unbalanced Sinkhorn algorithm [Chizat 2016] with KL-divergence terms for soft marginal constraints, with weights `(lambda1, lambda2)`
for the marginals mu, nu respectively.

Reference:
Chizat, L., Peyré, G., Schmitzer, B. and Vialard, F.X., 2016. Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.
"""
function sinkhorn_unbalanced(mu, nu, C, lambda1, lambda2, eps; tol = 1e-9, max_iter = 1000, verbose = false)
    function proxdiv_KL(s, eps, lambda, p)
        return @. (s^(eps/(eps + lambda)) * p^(lambda/(eps + lambda)))/s
    end

    a = ones(size(mu, 1)); b = ones(size(nu, 1))
    a_old = a; b_old = b
    tmp_a = zeros(size(nu, 1)); tmp_b = zeros(size(mu, 1))

    K = @. exp(-C/eps)

    iter = 1

    while true
        a_old = a
        b_old = b
        tmp_b = K*b
        a = proxdiv_KL(tmp_b, eps, lambda1, mu)
        tmp_a = K'*a
        b = proxdiv_KL(tmp_a, eps, lambda2, nu)
        iter += 1
        if iter % 10 == 0
            err_a = maximum(abs.(a - a_old))/max(maximum(abs.(a)), maximum(abs.(a_old)), 1)
            err_b = maximum(abs.(b - b_old))/max(maximum(abs.(b)), maximum(abs.(b_old)), 1)
            if verbose
                println("Iteration $iter, err = ", 0.5*(err_a + err_b))
            end
            if (0.5*(err_a + err_b) < tol) || iter > max_iter
                break
            end
        end
    end
    if iter > max_iter && verbose
        println( "Warning: exited before convergence")
    end
    return Diagonal(a)*K*Diagonal(b)
end


"""
    sinkhorn_unbalanced2(mu, nu, C, lambda1, lambda2, eps; kwargs...)

Computes the optimal transport cost of histograms `mu` and `nu` with cost matrix `C` and entropic regularization parameter `eps`, 
using the unbalanced Sinkhorn algorithm [Chizat 2016] with KL-divergence terms for soft marginal constraints, with weights `(lambda1, lambda2)`
for the marginals mu, nu respectively.
"""
function sinkhorn_unbalanced2(mu, nu, C, lambda1, lambda2, eps; kwargs...)
    return dot(C, sinkhorn_unbalanced(mu, nu, C, lambda1, lambda2, eps; kwargs...))
end

"""
    pot_sinkhorn(mu, nu, C, eps; kwargs...)

Compute optimal transport map of histograms `mu` and `nu` with cost matrix `C` and entropic
regularization parameter `eps`. 

Method can be a choice of `"sinkhorn"`, `"greenkhorn"`, `"sinkhorn_stabilized"`, or `"sinkhorn_epsilon_scaling"`.

This function is a wrapper of the function
[`sinkhorn`](https://pythonot.github.io/all.html?highlight=sinkhorn#ot.sinkhorn) in the
Python Optimal Transport package.
"""
function pot_sinkhorn(mu, nu, C, eps; tol=1e-9, max_iter = 1000, method = "sinkhorn", verbose = false)
    return pot.sinkhorn(nu, mu, PyReverseDims(C), eps; stopThr = tol, numItermax = max_iter, method = method, verbose = verbose)'
end

"""
    pot_sinkhorn2(mu, nu, C, eps; kwargs...)

Compute optimal transport cost of histograms `mu` and `nu` with cost matrix `C` and entropic
regularization parameter `eps`.

Method can be a choice of `"sinkhorn"`, `"greenkhorn"`, `"sinkhorn_stabilized"`, or `"sinkhorn_epsilon_scaling"`.

This function is a wrapper of the function
[`sinkhorn2`](https://pythonot.github.io/all.html?highlight=sinkhorn#ot.sinkhorn2) in the
Python Optimal Transport package.
"""
function pot_sinkhorn2(mu, nu, C, eps; tol=1e-9, max_iter = 1000, method = "sinkhorn", verbose = false)
    return pot.sinkhorn2(nu, mu, PyReverseDims(C), eps; stopThr = tol, numItermax = max_iter, method = method, verbose = verbose)[1]
end

"""
    pot_sinkhorn_unbalanced(mu, nu, C, eps, lambda; kwargs...)

Compute optimal transport map of histograms `mu` and `nu` with cost matrix `C`, using entropic regularisation parameter `eps` and marginal weighting functions `lambda`.

This function is a wrapper of the function
[`sinkhorn_unbalanced`](https://pythonot.github.io/all.html?highlight=sinkhorn_unbalanced#ot.sinkhorn_unbalanced) in the Python Optimal Transport package.
"""
function pot_sinkhorn_unbalanced(mu, nu, C, eps, lambda; tol = 1e-9, max_iter = 1000, method = "sinkhorn", verbose = false)
    return pot.sinkhorn_unbalanced(nu, mu, PyReverseDims(C), eps, lambda; stopThr = tol, numItermax = max_iter, method = method, verbose = verbose)'
end


"""
    pot_sinkhorn_unbalanced2(mu, nu, C, eps, lambda; kwargs...)

Compute optimal transport cost of histograms `mu` and `nu` with cost matrix `C`, using entropic regularisation parameter `eps` and marginal weighting functions `lambda`.

This function is a wrapper of the function
[`sinkhorn_unbalanced2`](https://pythonot.github.io/all.html#ot.sinkhorn_unbalanced2) in the Python Optimal Transport package.
"""
function pot_sinkhorn_unbalanced2(mu, nu, C, eps, lambda; tol = 1e-9, max_iter = 1000, method = "sinkhorn", verbose = false)

    return pot.sinkhorn_unbalanced2(nu, mu, PyReverseDims(C), eps, lambda; stopThr = tol, numItermax = max_iter, method = method, verbose = verbose)[1]
end

"""
    sinkhorn_stabilized_epsscaling(mu, nu, C, eps; kwargs...) 

Compute optimal transport map of histograms `mu` and `nu` with cost matrix `C` and entropic regularisation parameter `eps`. 
Uses stabilized Sinkhorn algorithm with epsilon-scaling [Schmitzer 2019]. 

`k` epsilon-scaling steps are used with scaling factor `lambda`, i.e. sequentially solve Sinkhorn with regularisation parameters 
`[lambda^(1-k), ..., lambda^(-1), 1]*eps`. 


Reference: 
Schmitzer, B., 2019. Stabilized sparse scaling algorithms for entropy regularized transport problems. SIAM Journal on Scientific Computing, 41(3), pp.A1443-A1481.
"""
function sinkhorn_stabilized_epsscaling(mu, nu, C, eps; absorb_tol = 1e3, max_iter = 1000, tol = 1e-9, lambda = 0.5, k = 5, verbose = false)
    eps_values = [eps*lambda^(k-j) for j = 1:k]
    alpha = zeros(size(mu)); beta = zeros(size(nu))
    for eps in eps_values
        if verbose; println(string("Warm start: eps = ", eps)); end
        alpha, beta = sinkhorn_stabilized(mu, nu, C, eps, absorb_tol = absorb_tol, max_iter = max_iter, tol = tol, alpha = alpha, beta = beta, return_duals = true, verbose = verbose)
    end
    K = exp.(-(C .- alpha .- beta')/eps).*mu.*nu'
    return K
end

function getK(C, alpha, beta, eps, mu, nu)
    return (exp.(-(C .- alpha .- beta')/eps).*mu.*nu')
end

"""
    sinkhorn_stabilized(mu, nu, C, eps; kwargs...) 
Compute optimal transport map of histograms `mu` and `nu` with cost matrix `C` and entropic regularisation parameter `eps`. 
Uses stabilized Sinkhorn algorithm [Schmitzer 2019].

Reference: 
Schmitzer, B., 2019. Stabilized sparse scaling algorithms for entropy regularized transport problems. SIAM Journal on Scientific Computing, 41(3), pp.A1443-A1481.
"""
function sinkhorn_stabilized(mu, nu, C, eps; absorb_tol = 1e3, max_iter = 1000, tol = 1e-9, alpha = nothing, beta = nothing, return_duals = false, verbose = false)
    if isnothing(alpha) || isnothing(beta)
        alpha = zeros(size(mu)); beta = zeros(size(nu))
    end

    u = ones(size(mu)); v = ones(size(nu))
    K = getK(C, alpha, beta, eps, mu, nu)
    i = 0

    if !(sum(mu) ≈ sum(nu))
        throw(ArgumentError("Error: mu and nu must lie in the simplex"))
    end

    while true
        u = mu./(K*v .+ 1e-16)
        v = nu./(K'*u .+ 1e-16)
        if (max(norm(u, Inf), norm(v, Inf)) > absorb_tol)
            if verbose; println("Absorbing (u, v) into (alpha, beta)"); end
            # absorb into α, β
            alpha = alpha + eps*log.(u); beta = beta + eps*log.(v)
            u = ones(size(mu)); v = ones(size(nu))
            K = getK(C, alpha, beta, eps, mu, nu)
        end
        if i % 10 == 0
            # check marginal
            gamma = getK(C, alpha, beta, eps, mu, nu).*(u.*v')
            err_mu = norm(gamma*ones(size(nu)) - mu, Inf)
            err_nu = norm(gamma'*ones(size(mu)) - nu, Inf)
            if verbose; println(string("Iteration ", i, ", err = ", 0.5*(err_mu + err_nu))); end
            if 0.5*(err_mu + err_nu) < tol
                break
            end
        end

        if i > max_iter
            if verbose; println("Warning: exited before convergence"); end
            break
        end
        i+=1
    end
    alpha = alpha + eps*log.(u); beta = beta + eps*log.(v)
    if return_duals
        return alpha, beta
    end
    return getK(C, alpha, beta, eps, mu, nu)
end



"""
SemiSmoothNewton(mu, nu, C, ϵ; kwargs...)

Computes the optimal transport map of histograms `mu` and `nu` with cost matrix `C` and quadratic regularization parameter `ϵ`, 
using the semismooth Newton algorithm [Lorenz 2016].

This implementation makes use of IterativeSolvers.jl and SparseArrays.jl.

Parameters:
θ: starting Armijo parameter.
tol: tolerance of marginal error.
maxiter: maximum interation number.
κ: control parameter of Armijo.
δ: small constant for the numerical stability of conjugate gradient iterative solver.

Tips:
If the algorithm does not converge, try some different values of θ.

Reference:
Lorenz, D.A., Manns, P. and Meyer, C., 2019. Quadratically regularized optimal transport. arXiv preprint arXiv:1903.01112v4.
"""
function SemiSmoothNewton(mu, nu, C, ϵ;        
    θ = 0.01,
    tol = 0.01,
    maxiter = 500,
    κ = 0.5,
    δ = 1e-5
    )
        
    

    if !(sum(mu) ≈ sum(nu))
        throw(ArgumentError("Error: mu and nu must lie in the simplex"))
    end

    N = length(mu)
    M = length(nu)

    # initialize dual potentials as uniforms
    a = ones(M)./M
    b = ones(N)./N

    γ = spzeros(M, N)

    converged = false

    function DualObjective(a, b)
        A = a.*ones(N)' + ones(M).*b' - C

        return 0.5 * norm(A[A.>0], 2) - ϵ*(dot(nu, a)+ dot(mu, b))
    end
    
    # computes minimizing directions
    function search_dir(a, b)
        
        P  = a*ones(N)' .+ ones(M)*b' .- C
        
        σ = 1.0*sparse(P .>= 0)
        γ = sparse(max.(P, 0)./ ϵ)
        
        G = vcat(
            hcat(spdiagm(0 => σ*ones(N)), σ), 
            hcat(sparse(σ'), 
            spdiagm(0 => sparse(σ')*ones(M) ))
            )
        
        h = vcat(
            γ*ones(N) - nu, 
            sparse(γ')*ones(M) - mu
            )
                    
        x = cg(G + δ*I, -ϵ * h)
        
        return x[1:M], x[M+1:end]
    end

    # computes optimal maginitude in the minimizing directions
    function search_t(a, b, da, db, θ)
        
        d =  ϵ * dot(γ , (da .* ones(N)' .+ ones(M) .* db')) -ϵ*(dot(da, nu) + dot(db, mu))
        
        ϕ₀ = DualObjective(a, b)
        t = 1
        
        while DualObjective(a+t*da, b+t*db) >= ϕ₀ + t*θ*d
            t *= κ
            
            if t < 1e-15
                # @warn "@ i = $i, t = $t , armijo did not converge"
                break
            end
            
        end
        return t
    end

    for i in 1:maxiter
        
        da, db = search_dir(a, b)
        
        t = search_t(a, b, da, db, θ)
        
        a += t*da
        b += t*db
        
        err1 = norm(γ*ones(N) -nu, Inf) 
        err2 = norm(sparse(γ')*ones(M) - mu, Inf)
        
        if err1 <= tol && err2 <= tol
            converged = true
            print("Converged @ i = $i with marginal errors: \n
            err1 = $err1, err2 = $err2 \n")
            break
        elseif i == maxiter
            print("Not Converged with errors:\n err1 = $err1, err2 = $err2 \n")
        end
        
        
    end

    if  !converged
        @warn "SemiSmooth Newton algorithm did not converge"    
    end

    return Matrix(γ')
end



end
