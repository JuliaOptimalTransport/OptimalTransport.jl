"""
    sinkhorn_barycenter(μ, C, ε, w; tol=1e-9, check_marginal_step=10, max_iter=1000)

Compute the Sinkhorn barycenter for a collection of `N` histograms contained in the columns of `μ`, for a cost matrix `C` of size `(size(μ, 1), size(μ, 1))`, relative weights `w` of size `N`, and entropic regularisation parameter `ε`.
Returns the entropically regularised barycenter of the `μ`, i.e. the histogram `ρ` of length `size(μ, 1)` that solves

```math
\\min_{\\rho \\in \\Sigma} \\sum_{i = 1}^N w_i \\operatorname{OT}_{\\varepsilon}(\\mu_i, \\rho)
```

where ``\\operatorname{OT}_{ε}(\\mu, \\nu) = \\inf_{\\gamma \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle + \\varepsilon \\Omega(\\gamma)``
is the entropic optimal transport loss with cost ``C`` and regularisation ``\\epsilon``.
"""
function sinkhorn_barycenter(μ, C, ε, w; tol=1e-9, check_marginal_step=10, max_iter=1000)
    sums = sum(μ; dims=1)
    if !isapprox(extrema(sums)...)
        throw(ArgumentError("Error: marginals are unbalanced"))
    end
    K = exp.(-C / ε)
    converged = false
    v = ones(size(μ))
    u = ones(size(μ))
    N = size(μ, 2)
    for n in 1:max_iter
        v = μ ./ (K' * u)
        a = ones(size(u, 1))
        a = prod((K * v)' .^ w; dims=1)'
        u = a ./ (K * v)
        if n % check_marginal_step == 0
            # check marginal errors
            err = maximum(abs.(μ .- v .* (K' * u)))
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
    return u[:, 1] .* (K * v[:, 1])
end
