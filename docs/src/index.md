# OptimalTransport.jl Documentation


## Exact optimal transport (Kantorovich) problem

OptimalTransport.jl reexports the following functions for exact, i.e.,
unregularized, optimal transport problems from
[ExactOptimalTransport.jl](https://github.com/JuliaOptimalTransport/ExactOptimalTransport.jl).

```@docs
emd
emd2
ot_plan
ot_plan(::Any, ::ExactOptimalTransport.ContinuousUnivariateDistribution, ::ExactOptimalTransport.UnivariateDistribution)
ot_plan(::Any, ::ExactOptimalTransport.DiscreteNonParametric, ::ExactOptimalTransport.DiscreteNonParametric)
ot_plan(::ExactOptimalTransport.SqEuclidean, ::ExactOptimalTransport.Normal, ::ExactOptimalTransport.Normal)
ot_plan(::ExactOptimalTransport.SqEuclidean, ::ExactOptimalTransport.MvNormal, ::ExactOptimalTransport.MvNormal)
ot_cost
ot_cost(::Any, ::ExactOptimalTransport.ContinuousUnivariateDistribution, ::ExactOptimalTransport.UnivariateDistribution)
ot_cost(::Any, ::ExactOptimalTransport.DiscreteNonParametric, ::ExactOptimalTransport.DiscreteNonParametric)
ot_cost(::ExactOptimalTransport.SqEuclidean, ::ExactOptimalTransport.Normal, ::ExactOptimalTransport.Normal)
ot_cost(::ExactOptimalTransport.SqEuclidean, ::ExactOptimalTransport.MvNormal, ::ExactOptimalTransport.MvNormal)
wasserstein
squared2wasserstein
discretemeasure
```

## Entropically regularised optimal transport

```@docs
sinkhorn
sinkhorn2
sinkhorn_divergence
sinkhorn_barycenter
```

Currently the following variants of the Sinkhorn algorithm are supported:

```@docs
SinkhornGibbs
SinkhornStabilized
SinkhornEpsilonScaling
Greenkhorn
```

The following methods are deprecated and will be removed:

```@docs
sinkhorn_stabilized
sinkhorn_stabilized_epsscaling
```

## Unbalanced optimal transport

```@docs
sinkhorn_unbalanced
sinkhorn_unbalanced2
sinkhorn_divergence_unbalanced
```

## Quadratically regularised optimal transport

```@docs
quadreg
```

Currently the following algorithms for solving quadratically regularised optimal transport are supported:
```@docs
QuadraticOTNewton
```

## Dual

```@docs
OptimalTransport.Dual.ot_entropic_semidual
OptimalTransport.Dual.ot_entropic_semidual_grad
OptimalTransport.Dual.getprimal_ot_entropic_semidual
OptimalTransport.Dual.ot_entropic_dual
OptimalTransport.Dual.ot_entropic_dual_grad
OptimalTransport.Dual.getprimal_ot_entropic_dual
```
