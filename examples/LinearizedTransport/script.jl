"""
Optimal transport costs are expensive to compute in general, so scaling can be quite bad if we need to, say, compute
the OT cost pairwise for a reasonably sized family of measures. When this is the situation, it may be beneficial to
linearize the OT distance using the manifold-like structure induced by the Wasserstein cost. Fix μ, and consider the transformation
ν → T_ν, where T_ν is the optimal transport map pushing μ forward to ν. Now fix two other measures ν, ρ, not equal to μ.
We may approximate OT(ν, ρ) via OT(ν, ρ) ≈ ||T_ν - T_ρ||_L^2(μ). If μ, ν, and ρ are "nice" (i.e. have smooth and accessible densities
w.r.t to the Lebesgue measure), then the right hand side is easy to approximate well via standard numerical methods.

Now, it is a sad fact that recovering the maps T_ν is generally no easy task itself. But in the case of entropically regularized
transport, there exists a very nice entropic approximation to the transport map, which depends only on the measure ν and
a family of N i.i.d samples Y_i ∼ ν.

The following example is rather contrived, since if we only wanted to compute one distance, we're actually doing much more work than we
need to by computing 2 Sinkhorn problems and an integral on top of that, but again the main application here would be when
we have O(n^2) distances to compute

Note that the choice of reference measure can significantly affect the quality of the approximation, and as of writing there is
no non-heauristic method for choosing a "good" reference.

Relevant sources:

Moosmüller, Caroline, and Alexander Cloninger. "Linear optimal transport embedding: provable Wasserstein classification for certain rigid transformations and perturbations." Information and Inference: A Journal of the IMA 12.1 (2023): 363-389.
Pooladian, A.-A. and Niles-Weed, J. Entropic estimation of optimal transport maps. arXiv: 2109.12004, 2021

"""

using Distances
using Distributions
using OptimalTransport

N = 1000 # number of samples

# sample some points according to our chosen reference and target distributions
μ = rand(Normal(1,1), N)
ν = rand(Normal(0,1), N)
ρ = rand(Normal(2,1), N)

# set the weights on the samples to be uniform
a = fill(1/N, N)

# compute the cost matrices for the two pairs of distributions
C = pairwise(SqEuclidean(), μ', ν')
D = pairwise(SqEuclidean(), μ', ρ')
E = pairwise(SqEuclidean(), ν', ρ')

# get the entropic transport maps
T_ν = entropic_transport_map(a, a, ν, C, 0.1, SinkhornGibbs())
T_ρ = entropic_transport_map(a, a, ρ, D, 0.1, SinkhornGibbs())

# integrand for the linearization
f(x) = (T_ν([x]) - T_ρ([x]))^2

# convert target distributions to dirac clouds
ν_dist = DiscreteNonParametric(ν, a)
ρ_dist = DiscreteNonParametric(ρ, a)

# compute and compare
I = (sum(f.(μ)) /  N)^0.5 # naive Monte Carlo approximation of the L2 distance between the entropic maps
J = ot_cost(sqeuclidean, ν_dist, ρ_dist)

println("Linear approximation of the distance: $I; True OT distance: $J")
