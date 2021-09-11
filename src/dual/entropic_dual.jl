module Dual

using LinearAlgebra
using LogExpFunctions
import OptimalTransport.add_singleton
import OptimalTransport.dot_vecwise

export ot_entropic_semidual
export ot_entropic_semidual_grad
export getprimal_ot_entropic_semidual
export ot_entropic_dual
export ot_entropic_dual_grad
export getprimal_ot_entropic_dual

function logKexp(logK::AbstractMatrix, x::AbstractVector)
    return logsumexp(logK .+ add_singleton(x, Val(1)); dims=2)
end

function logKexp(logK::AbstractMatrix, x::AbstractMatrix; dims)
    return logsumexp(logK + x; dims=dims)
end

"""
    ot_entropic_semidual(μ, v, eps, K)

Computes the semidual (in the second argument) of the entropic optimal transport loss, with source marginal `μ`, regularization parameter `ε`, and Gibbs kernel `K`.  

That is, 
```math
    \\operatorname{OT}_{\\varepsilon}(\\mu, \\nu) = \\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle + \\varepsilon \\Omega(\\gamma)
```
with ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{ij} \\log \\gamma_{ij}``, then the semidual in the second argument ν is [^Z21]
```math
\\begin{aligned}
    \\operatorname{OT}_{\\varepsilon}^*(\\mu, v) &= \\sup_{\\nu} \\langle v, \\nu \\rangle - \\operatorname{OT}_{\\varepsilon}(\\mu, \\nu) \\ \\ 
    &= -\\varepsilon \\left\\langle \\mu, \\log\\left( \\dfrac{\\mu}{K e^{v/\\varepsilon}} \\right) - 1\\right\\rangle. 
\\end{aligned}
```
Notably, the semidual is computationally advantageous for solving variational problems since it is a smooth and unconstrained function of `v` since it admits a closed form gradient. See [^CP16] for a detailed discussion of dual methods for variational problems in optimal transport. 

[^CP16]: Cuturi, Marco, and Gabriel Peyré. "A smoothed dual approach for variational Wasserstein problems." SIAM Journal on Imaging Sciences 9.1 (2016): 320-343.
[^Z21]: Zhang, Stephen Y. “A Unified Framework for Non-Negative Matrix and Tensor Factorisations with a Smoothed Wasserstein Loss.” ArXiv: Machine Learning, 2021.
"""
function ot_entropic_semidual(μ, v, eps, K)
    return eps *
           (-dot_vecwise(xlogx.(μ) .- μ, one.(μ)) + dot_vecwise(μ, log.(K * exp.(v / eps))))
end

"""
    ot_entropic_semidual_grad(μ, v, eps, K)

Computes the gradient with respect to `v` of the semidual of the entropic optimal transport loss. That is,
```math
\\nabla_v \\operatorname{OT}^*_{\\varepsilon}(\\mu, v) = K^\\top \\left( \\dfrac{\\mu}{K e^{v/\\varepsilon}} \\right) \\odot e^{v/\\varepsilon}.
```

See also: [`ot_entropic_semidual`](@ref)
"""
function ot_entropic_semidual_grad(μ, v, eps, K)
    return K' * (μ ./ (K * exp.(v / eps))) .* exp.(v / eps)
end

"""
    getprimal_ot_entropic_semidual(μ, v, eps, K)

Computes the the primal variable `ν` corresponding to the dual variable `v` at optimality. That is,
```math
\\nu^\\star = e^{v^\\star/\\varepsilon} \\odot K^\\top \\dfrac{\\mu}{K e^{v^\\star/\\varepsilon}}. 
```

See also: [`ot_entropic_semidual`](@ref)
"""
function getprimal_ot_entropic_semidual(μ, v, eps, K)
    return exp.(v / eps) .* (K' * (μ ./ (K * exp.(v / eps))))
end

"""
    ot_entropic_dual(u, v, eps, K)

Computes the dual in both arguments of entropic optimal transport loss, where `u` and `v` are the dual variables associated with the source and target marginals respectively. 

That is,
```math
    \\begin{aligned}
    \\operatorname{OT}_{\\varepsilon}^*(u, v) &= \\sup_{\\mu, \\nu} \\langle u, \\mu \\rangle + \\langle v, \\nu \\rangle - \\operatorname{OT}_\\varepsilon(\\mu, \\nu) \\ \\ 
    &= \\varepsilon \\log \\langle e^{u/\\varepsilon}, K e^{v/\\varepsilon} \\rangle. 
    \\end{aligned}
```
"""
function ot_entropic_dual(u, v, eps, K)
    # (μ, ν) → min_{γ ∈ Π(μ, ν)} ε H(γ | K)
    # has Legendre transform
    # (u, v) → ε log < exp(u/ε), K exp(v/ε) >
    return eps * log.(dot_vecwise(exp.(u / eps), K * exp.(v / eps)))
end

"""
    ot_entropic_dual_grad(u, v, eps, K)

Computes the gradient with respect to `u` and `v` of the dual of the entropic optimal transport loss. That is,
```math
\\begin{aligned}
\\nabla_u \\operatorname{OT}^*_{\\varepsilon}(u, v) &= \\dfrac{e^{u/\\varepsilon} \\odot K e^{v/\\varepsilon}}{\\langle e^{u/\\varepsilon}, K e^{v/\\varepsilon} \\rangle} \\ \\
\\nabla_v \\operatorname{OT}^*_{\\varepsilon}(u, v) &= \\dfrac{e^{v/\\varepsilon} \\odot K^\\top e^{u/\\varepsilon}}{\\langle e^{v/\\varepsilon}, K^\\top e^{u/\\varepsilon} \\rangle}.
\\end{aligned}
```

See also: [`ot_entropic_dual`](@ref)
"""
function ot_entropic_dual_grad(u, v, eps, K)
    U = exp.(u / eps)
    V = exp.(v / eps)
    grad_u = (U .* (K * V)) ./ dot_vecwise(U, K * V)'
    grad_v = (V .* (K' * U)) ./ dot_vecwise(V, K' * U)'
    return grad_u, grad_v
end

"""
    getprimal_ot_entropic_dual(u, v, eps, K)

Computes the the primal variable `γ` corresponding to the dual variable `u, v` at optimality. That is,
```math
    \\gamma = \\operatorname{softmax}(\\mathrm{diag}(e^{u/\\varepsilon}) K \\mathrm{diag}(e^{v/\\varepsilon}))
```

See also: [`ot_entropic_dual`](@ref)
"""
function getprimal_ot_entropic_dual(u, v, eps, K)
    γ = K .* add_singleton(exp.(-u / eps), Val(2)) .* add_singleton(exp.(-v / eps), Val(1))
    return γ
end

end
