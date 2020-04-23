# Julia optimal transport routines
# Author: Stephen Zhang (syz@math.ubc.ca)


# ENV["PYTHON"] = "/home/zsteve/anaconda3/bin/python"
Pkg.build("PyCall")
using PyCall
pot = pyimport("ot")

function emd(a, b, M)
    """
    Exact solution to Kantorovich problem. Return optimal transport coupling \gamma.
    """
    return pot.lp.emd(b, a, PyReverseDims(M))'
end

function emd2(a, b, M)
    """
    Exact solution to Kantorovich problem. Retern optimal transport cost \int c d\gamma. 
    """
    return pot.lp.emd2(b, a, PyReverseDims(M))
end

function sinkhorn_impl(mu, nu, C, eps; tol = 1e-6, check_marginal_step = 10, max_iter = 1000, verbose = false)
    """
    Sinkhorn algorithm to compute coupling of mu, nu with regularisation eps. 
    Return dual potentials u, v such that \gamma_ij = u_i K_ij v_j.
    """
    K = exp.(-C/eps)
    v = ones(size(C, 2))
    u = ones(size(C, 1))
    temp_v = zeros(size(C, 1))
    temp_u = zeros(size(C, 2))
    iter = 0
    while true
        mul!(temp_v, K, v)
        mul!(u, Diagonal(1 ./temp_v), mu)
        mul!(temp_u, K', u)
        mul!(v, Diagonal(1 ./temp_u), nu)
        # check mu marginal
        if (iter > max_iter) || ((iter % check_marginal_step == 0) && (maximum(abs.(mu - u.*(K*v))) < tol))
            if iter > max_iter && verbose
                println("Warning: sinkhorn_native exited without converging")
            end
            break
        end
        count += 1
    end
    return u, v
end

function sinkhorn(mu, nu, C, eps; tol = 1e-6, check_marginal_step = 10, max_iter = 1000)
    """
    Sinkhorn algorithm for entropically regularised optimal transport.
    Return optimal transport coupling \gamma.
    """
    u, v = sinkhorn_impl(mu, nu, C, eps;
                        tol = tol, check_marginal_step = check_marginal_step, max_iter = max_iter)
    return Diagonal(u)*exp.(-C/eps)*Diagonal(v)
end

function sinkhorn2(mu, nu, C, eps; tol = 1e-6, check_marginal_step = 10, max_iter = 1000)
    """
    Sinkhorn algorithm for entropically regularised optimal transport.
    Return optimal transport cost \int c d\gamma + \eps H(\gamma | \mu \otimes \nu)
    """
    gamma = sinkhorn(mu, nu, C, eps;
                        tol = tol, check_marginal_step = check_marginal_step2, max_iter = max_iter)
    return sum(gamma.*C)
end

function sinkhorn_unbalanced(μ, ν, C, λ1, λ2, ϵ; tol = 1e-6, max_iter = 1000, verbose = false)
    """
    Unbalanced Sinkhorn algorithm with KL (\lambda_1, \lambda_2) marginal terms for marginals 
    \mu, \nu respectively.
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
