# OptimalTransport.jl -- optimal transportation algorithms for Julia
# See prettyprinted documentation at https://juliaoptimaltransport.github.io/OptimalTransport.jl/dev/
#

module OptimalTransport

using Reexport

@reexport using ExactOptimalTransport:
    emd, emd2, ot_cost, ot_plan, wasserstein, squared2wasserstein, discretemeasure

using LinearAlgebra
using IterativeSolvers
using LogExpFunctions: LogExpFunctions
using NNlib: NNlib

export SinkhornGibbs, SinkhornStabilized, SinkhornEpsilonScaling
export SinkhornBarycenterGibbs
export QuadraticOTNewton

export sinkhorn, sinkhorn2
export sinkhorn_stabilized, sinkhorn_stabilized_epsscaling, sinkhorn_barycenter
export sinkhorn_unbalanced, sinkhorn_unbalanced2
export sinkhorn_divergence
export quadreg

include("utils.jl")

include("entropic/sinkhorn.jl")
include("entropic/sinkhorn_divergence.jl")
include("entropic/sinkhorn_gibbs.jl")
include("entropic/symmetric.jl")
include("entropic/sinkhorn_stabilized.jl")
include("entropic/sinkhorn_epsscaling.jl")
include("entropic/sinkhorn_unbalanced.jl")
include("entropic/sinkhorn_barycenter.jl")
include("entropic/sinkhorn_barycenter_gibbs.jl")
include("entropic/sinkhorn_solve.jl")

include("quadratic.jl")
include("quadratic_newton.jl")

include("dual/entropic_dual.jl")

end
