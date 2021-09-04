# OptimalTransport.jl -- optimal transportation algorithms for Julia
# See prettyprinted documentation at https://juliaoptimaltransport.github.io/OptimalTransport.jl/dev/
#

module OptimalTransport

using Distances
using LinearAlgebra
using IterativeSolvers, SparseArrays
using LogExpFunctions: LogExpFunctions
using MathOptInterface
using Distributions
using PDMats
using QuadGK
using NNlib: NNlib
using StatsBase: StatsBase
using Optim

export SinkhornGibbs, SinkhornStabilized, SinkhornEpsilonScaling
export SinkhornBarycenterGibbs
export QuadraticOTNewton

export sinkhorn, sinkhorn2
export emd, emd2
export sinkhorn_stabilized, sinkhorn_stabilized_epsscaling, sinkhorn_barycenter
export sinkhorn_unbalanced, sinkhorn_unbalanced2
export quadreg
export ot_cost, ot_plan, wasserstein, squared2wasserstein

const MOI = MathOptInterface

include("distances/bures.jl")
include("utils.jl")
include("exact.jl")
include("wasserstein.jl")

include("entropic/sinkhorn.jl")
include("entropic/sinkhorn_gibbs.jl")
include("entropic/sinkhorn_stabilized.jl")
include("entropic/sinkhorn_epsscaling.jl")
include("entropic/sinkhorn_unbalanced.jl")
include("entropic/sinkhorn_barycenter.jl")
include("entropic/sinkhorn_barycenter_gibbs.jl")
include("entropic/sinkhorn_solve.jl")

include("quadratic.jl")
include("quadratic_newton.jl")

include("dual/entropic_dual.jl")
include("entropic/sinkhorn_dual.jl")

end
