module POT

using ..PyCall

const pot = PyNULL()

function __init__()
    copy!(pot, pyimport_conda("ot", "pot", "conda-forge"))
end

"""
    emd(mu, nu, C)

Compute transport map for Monge-Kantorovich problem with source and target marginals `mu`
and `nu` and a cost matrix `C` of dimensions `(length(mu), length(nu))`.

Return optimal transport coupling `γ` of the same dimensions as `C` which solves 

```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle
```

This function is a wrapper of the function
[`emd`](https://pythonot.github.io/all.html#ot.emd) in the Python Optimal Transport
package.
"""
function emd(mu, nu, C)
    return pot.lp.emd(nu, mu, PyReverseDims(C))'
end

"""
    emd2(mu, nu, C)

Compute exact transport cost for Monge-Kantorovich problem with source and target marginals
`mu` and `nu` and a cost matrix `C` of dimensions `(length(mu), length(nu))`.

Returns optimal transport cost (a scalar), i.e. the optimal value

```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle
```

This function is a wrapper of the function
[`emd2`](https://pythonot.github.io/all.html#ot.emd2) in the Python Optimal Transport
package.
"""
function emd2(mu, nu, C)
    return pot.lp.emd2(nu, mu, PyReverseDims(C))[1]
end

"""
    sinkhorn(mu, nu, C, eps; tol=1e-9, max_iter = 1000, method = "sinkhorn", verbose = false)

Compute optimal transport map of histograms `mu` and `nu` with cost matrix `C` and entropic
regularization parameter `eps`. 

Method can be a choice of `"sinkhorn"`, `"greenkhorn"`, `"sinkhorn_stabilized"`, or
`"sinkhorn_epsilon_scaling"` (Flamary et al., 2017).

This function is a wrapper of the function
[`sinkhorn`](https://pythonot.github.io/all.html?highlight=sinkhorn#ot.sinkhorn) in the
Python Optimal Transport package.
"""
function sinkhorn(mu, nu, C, eps; tol=1e-9, max_iter = 1000, method = "sinkhorn", verbose = false)
    return pot.sinkhorn(nu, mu, PyReverseDims(C), eps; stopThr = tol, numItermax = max_iter, method = method, verbose = verbose)'
end

"""
    sinkhorn2(mu, nu, C, eps; tol=1e-9, max_iter = 1000, method = "sinkhorn", verbose = false)

Compute optimal transport cost of histograms `mu` and `nu` with cost matrix `C` and
entropic regularization parameter `eps`.

Method can be a choice of `"sinkhorn"`, `"greenkhorn"`, `"sinkhorn_stabilized"`, or
`"sinkhorn_epsilon_scaling"` (Flamary et al., 2017).

This function is a wrapper of the function
[`sinkhorn2`](https://pythonot.github.io/all.html?highlight=sinkhorn#ot.sinkhorn2) in the
Python Optimal Transport package.
"""
function sinkhorn2(mu, nu, C, eps; tol=1e-9, max_iter = 1000, method = "sinkhorn", verbose = false)
    return pot.sinkhorn2(nu, mu, PyReverseDims(C), eps; stopThr = tol, numItermax = max_iter, method = method, verbose = verbose)[1]
end

"""
    sinkhorn_unbalanced(mu, nu, C, eps, lambda; tol = 1e-9, max_iter = 1000, method = "sinkhorn", verbose = false)

Compute optimal transport map of histograms `mu` and `nu` with cost matrix `C`, using
entropic regularisation parameter `eps` and marginal weighting functions `lambda`.

This function is a wrapper of the function
[`sinkhorn_unbalanced`](https://pythonot.github.io/all.html?highlight=sinkhorn_unbalanced#ot.sinkhorn_unbalanced)
in the Python Optimal Transport package.
"""
function sinkhorn_unbalanced(mu, nu, C, eps, lambda; tol = 1e-9, max_iter = 1000, method = "sinkhorn", verbose = false)
    return pot.sinkhorn_unbalanced(nu, mu, PyReverseDims(C), eps, lambda; stopThr = tol, numItermax = max_iter, method = method, verbose = verbose)'
end

"""
    sinkhorn_unbalanced2(mu, nu, C, eps, lambda; tol = 1e-9, max_iter = 1000, method = "sinkhorn", verbose = false)

Compute optimal transport cost of histograms `mu` and `nu` with cost matrix `C`, using
entropic regularisation parameter `eps` and marginal weighting functions `lambda`.

This function is a wrapper of the function
[`sinkhorn_unbalanced2`](https://pythonot.github.io/all.html#ot.sinkhorn_unbalanced2) in
the Python Optimal Transport package.
"""
function sinkhorn_unbalanced2(mu, nu, C, eps, lambda; tol = 1e-9, max_iter = 1000, method = "sinkhorn", verbose = false)
    return pot.sinkhorn_unbalanced2(nu, mu, PyReverseDims(C), eps, lambda; stopThr = tol, numItermax = max_iter, method = method, verbose = verbose)[1]
end

end
