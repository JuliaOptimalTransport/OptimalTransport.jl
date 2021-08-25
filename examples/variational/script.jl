# # Variational problems 
#
# In this example, we will numerically simulate an entropy-regularised Wasserstein gradient flow 
# approximating the Fokker-Planck and porous medium equations. 
# 
# The connection between Wasserstein gradient flows and (non)-linear PDEs is due to Jordan, Kinderlehrer and Otto. 
#
using OptimalTransport
using StatsBase, Distances 
using ReverseDiff, Optim, LinearAlgebra
using Plots
using BenchmarkTools 
using LogExpFunctions
using LaTeXStrings

# Here, we set up the computational domain that we work on - we discretize the interval $[-1, 1]$. 

support = LinRange(-1, 1, 64)
C = pairwise(SqEuclidean(), support')

# # Fokker-Planck equation as $W_2$ gradient flow
# For a potential function $\Psi$ and noise level $\sigma^2$, the Fokker-Planck equation is 
# ```math
# \partial_t \rho &= \nabla \cdot (\rho \nabla \Psi) + \frac{\sigma^2}{2} \Delta \rho. 
# ```
# The result of Jordan, Kinderlehrer and Otto (commonly referred to as the JKO theorem) states that 
# $\rho$ evolves following the 2-Wasserstein gradient of the Gibbs free energy functional
# ```math
#   F(\rho) &= \int \Psi d\rho + \int \log(\rho) d\rho. 
# ```
#
# A numerical scheme for computing this gradient flow is to solve
# ```math
#   \rho_{k + 1} = \operatorname{argmin}_{\rho} \operatorname{OT}_\varepsilon(\rho_k, \rho) + \tau F(\rho). 
# ```
#
# # Numerical setup 
# Now we set up various functionals that we will use.
# Generalised entropy: for $m = 1$ this is the relative entropy, and $m = 2$ this is squared L2. 
function E(ρ; m = 1) 
    if m == 1
        return sum(xlogx.(ρ)) - sum(ρ)
    elseif m > 1
        return dot(ρ, @. (ρ^(m-1.0) - m)/(m-1.0))
    end
end

# Now define $V(x)$ to be a potential energy function that has two potential wells at $x = ± 0.5$. 

V(x) = 10*(x-0.5)^2*(x+0.5)^2

# Having defined $V$, this induces a potential energy functional $\Psi$ on probability distributions $\rho$:
# ```math
#    \Psi(\rho) &= \int V(x) \rho(x) dx = \langle \psi, \rho \rangle . 
# ```

Ψ = V.(support)

# Define the time step $\tau$ and entropic regularisation level $\varepsilon$, and form the associated Gibbs kernel $K = e^{-C/\varepsilon}$. 

τ = 0.05
ε = 0.01
K = @. exp(-C/ε)

# We define the (non-smooth) initial condition $\rho_0$ in terms of step functions. 

H(x) = x > 0
ρ0 = @. H(support + 0.25) - H(support - 0.25) 
ρ0 = ρ0/sum(ρ0)

f_primal_fp(ρ, ρ0, τ, ε, C) = sinkhorn2(ρ, ρ0, C, ε; regularization = true) + τ*( dot(Ψ, ρ) + E(ρ) )
function step(ρ0, τ, ε, C)
    opt = optimize(u -> f_primal_fp(softmax(u), ρ0, τ, ε, C), ones(size(ρ0)), LBFGS(), Optim.Options(iterations = 50, g_tol = 1e-6); autodiff = :forward)
    return softmax(Optim.minimizer(opt))
end

# set up 
N = 10 
ρ = zeros(size(ρ0, 1), N)
ρ[:, 1] = ρ0
for i = 2:N
    @info "timestep i = $i"
    ρ[:, i] = step(ρ[:, i-1], τ, ε, C)
end

colors = range(colorant"red", stop = colorant"blue", length = N)
plot(support, ρ, title = L"F(\rho) = \langle V, \rho \rangle + \langle \rho, \log(\rho) \rangle"; palette = colors, legend = nothing)

# Porous medium equation via primal
f_primal_pme(ρ, ρ0, τ, ε, C) = sinkhorn2(ρ, ρ0, C, ε; regularization = true) + τ*(dot(Ψ, ρ) + E(ρ; m = 2))

function step(ρ0, τ, ε, C)
    opt = optimize(u -> f_primal_pme(softmax(u), ρ0, τ, ε, C), ones(size(ρ0)), LBFGS(), Optim.Options(iterations = 250, g_tol = 1e-6); autodiff = :forward)
    return softmax(Optim.minimizer(opt))
end

# set up 
N = 10 
ρ = zeros(size(ρ0, 1), N)
ρ[:, 1] = ρ0
@time begin
for i = 2:N
    @info "timestep i = $i"
    ρ[:, i] = step(ρ[:, i-1], τ, ε, C)
end
end

# plot 
plot(support, ρ, title = L"F(\rho)"; palette = colors, legend = nothing)
