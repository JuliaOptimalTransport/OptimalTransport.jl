# # Sinkhorn divergences 
# 
# In this tutorial we provide a minimal example for using the Sinkhorn divergence as a loss function [FSV+19] on empirical distributions. 
# [FSV+19]: Feydy, Jean, et al. "Interpolating between optimal transport and MMD using Sinkhorn divergences." The 22nd International Conference on Artificial Intelligence and Statistics. PMLR, 2019.
# 
# While entropy-regularised optimal transport $\operatorname{OT}_{\varepsilon}(\cdot, \cdot)$ is commonly used as a loss function, it suffers from a problem of *bias*: namely that $\nu \mapsto \operatorname{OT}_{\varepsilon}(\mu, \nu)$ is *not* minimised at $\nu = \mu$. 
#
# A fix to this problem is proposed by Genevay et al [GPC18] and subsequently Feydy et al. [FSV+19], which introduce the *Sinkhorn divergence* between two measures $\mu$ and $\nu$, defined as 
# ```math
# \operatorname{S}_{\varepsilon}(\mu, \nu) = \operatorname{OT}_{\varepsilon}(\mu, \nu) - \frac{1}{2} \operatorname{OT}_{\varepsilon}(\mu, \mu) - \frac{1}{2} \operatorname{OT}_{\varepsilon}(\nu, \nu).
# ```
# In the above, we have followed the convention taken by Feydy et al. and included the entropic regularisation in the definition of $\operatorname{OT}_\varepsilon$. 
# [GPC18]: Aude Genevay, Gabriel Peyré, Marco Cuturi, Learning Generative Models with Sinkhorn Divergences, Proceedings of the Twenty-First International Conference on Artficial Intelligence and Statistics, (AISTATS) 21, 2018
# [FSV+19]: Feydy, Jean, et al. "Interpolating between optimal transport and MMD using Sinkhorn divergences." The 22nd International Conference on Artificial Intelligence and Statistics. PMLR, 2019.
#
# Like the Sinkhorn loss, the Sinkhorn divergence is smooth and convex in both of its arguments. However, the Sinkhorn divergence is unbiased -- i.e. $S_{\varepsilon}(\mu, \nu) = 0$ iff $\mu = \nu$. 
#
# Unlike previous examples, here we demonstrate a learning problem similar to Figure 1 of Feydy et al. over *empirical measures*, i.e. measures that have the form $\mu = \frac{1}{N} \sum_{i = 1}^{N} \delta_{x_i}$ where $\delta_x$ is the Dirac delta function at $x$. 
# 
# We first load packages. 
#
using OptimalTransport
using ReverseDiff
using Distributions
using LinearAlgebra
using Distances
using Plots
using Logging
using Optim

# As a ground truth distribution, we set $\rho$ to be a Gaussian mixture model with `k = 5` components, equally spaced around a circle, and sample an empirical distribution of size $N$, $\mu \sim \rho$.

k = 3
d = 2
θ = π * range(0, 2(1 - 1 / k); length=k)
μ = 2 * hcat(sin.(θ), cos.(θ))
ρ = MixtureModel(MvNormal[MvNormal(x, 0.25 * I) for x in eachrow(μ)])
N = 100
μ_spt = rand(ρ, N)'
scatter(μ_spt[:, 1], μ_spt[:, 2]; markeralpha=0.25, title=raw"$\mu$")

# Now, suppose we want to approximate $\mu$ with another empirical distribution $\nu$, i.e. we want to minimise $\nu \mapsto \operatorname{S}_{\varepsilon}(\mu, \nu)$ over possible empirical distributions $\nu$. In this case we have $M$ particles in $\nu$, which we initialise following a Gaussian distribution. 
M = 100
ν_spt = rand(M, d);
# Assign uniform weights to the Diracs in each empirical distribution. 
μ = fill(1 / N, N)
ν = fill(1 / M, M);

# Since $\mu$ is fixed, we pre-compute the cost matrix $C_{\mu}$. 
C_μ = pairwise(SqEuclidean(), μ_spt');

# Define the loss function to minimise, where `x` specifies the locations of the Diracs in $\nu$. 
# 
# We will be using `ReverseDiff` with a precompiled tape. For this reason, we need the Sinkhorn algorithm to perform a fixed number of (e.g. 50) iterations.
# Currently, this can be achieved by setting `maxiter = 50` and `atol = rtol = 0` in calls to `sinkhorn` and `sinkhorn_divergence`. 
function loss(x, ε)
    C_μν = pairwise(SqEuclidean(), μ_spt', x')
    C_ν = pairwise(SqEuclidean(), x')
    return sinkhorn_divergence(μ, ν, C_μν, C_μ, C_ν, ε; maxiter=50, atol=rtol = 0, regularization = true)
end
# Set entropy regularisation parameter
ε = 1.0;

# Use ReverseDiff with a precompiled tape and Optim.jl to minimise $\nu \mapsto \operatorname{S}_{\varepsilon}(\mu, \nu)$. Note that this is problem is *not* convex, so we find a local minimium. 
const loss_tape = ReverseDiff.GradientTape(x -> loss(x, ε), ν_spt)
const compiled_loss_tape = ReverseDiff.compile(loss_tape)
opt = with_logger(SimpleLogger(stderr, Logging.Error)) do
    optimize(
        x -> loss(x, ε),
        (∇, x) -> ReverseDiff.gradient!(∇, compiled_loss_tape, x),
        ν_spt,
        GradientDescent(),
        Optim.Options(; iterations=10, g_tol=1e-6, show_trace=true),
    )
end
ν_opt = Optim.minimizer(opt)
plt1 = scatter(μ_spt[:, 1], μ_spt[:, 2]; markeralpha=0.25, title="Sinkhorn divergence")
scatter!(plt1, ν_opt[:, 1], ν_opt[:, 2]);

# For comparison, let us do the same computation again, but this time we want to minimise $\nu \mapsto \operatorname{OT}_{\varepsilon}(\mu, \nu)$. 
function loss_biased(x, ε)
    C_μν = pairwise(SqEuclidean(), μ_spt', x')
    return sinkhorn2(μ, ν, C_μν, ε; maxiter=50, atol=rtol = 0, regularization = true)
end
const loss_biased_tape = ReverseDiff.GradientTape(x -> loss_biased(x, ε), ν_spt)
const compiled_loss_biased_tape = ReverseDiff.compile(loss_biased_tape)
opt_biased = with_logger(SimpleLogger(stderr, Logging.Error)) do
    optimize(
        x -> loss_biased(x, ε),
        (∇, x) -> ReverseDiff.gradient!(∇, compiled_loss_biased_tape, x),
        ν_spt,
        GradientDescent(),
        Optim.Options(; iterations=10, g_tol=1e-6, show_trace=true),
    )
end
ν_opt_biased = Optim.minimizer(opt_biased)
plt2 = scatter(μ_spt[:, 1], μ_spt[:, 2]; markeralpha=0.25, title="Sinkhorn loss")
scatter!(plt2, ν_opt_biased[:, 1], ν_opt_biased[:, 2]);

# Observe that the Sinkhorn divergence results in $\nu$ that matches $\mu$ quite well, while entropy-regularised transport is biased to producing $\nu$ that seems to concentrate around the mean of each Gaussian component. 
plot(plt1, plt2)
