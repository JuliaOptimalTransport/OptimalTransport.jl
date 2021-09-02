# OptimalTransport.jl Documentation


## Exact optimal transport (Kantorovich) problem

```@docs
emd
emd2
ot_plan
ot_plan(::Any, ::OptimalTransport.ContinuousUnivariateDistribution, ::OptimalTransport.UnivariateDistribution)
ot_plan(::Any, ::OptimalTransport.DiscreteNonParametric, ::OptimalTransport.DiscreteNonParametric)
ot_plan(::OptimalTransport.SqEuclidean, ::OptimalTransport.Normal, ::OptimalTransport.Normal)
ot_plan(::OptimalTransport.SqEuclidean, ::OptimalTransport.MvNormal, ::OptimalTransport.MvNormal)
ot_cost
ot_cost(::Any, ::OptimalTransport.ContinuousUnivariateDistribution, ::OptimalTransport.UnivariateDistribution)
ot_cost(::Any, ::OptimalTransport.DiscreteNonParametric, ::OptimalTransport.DiscreteNonParametric)
ot_cost(::OptimalTransport.SqEuclidean, ::OptimalTransport.Normal, ::OptimalTransport.Normal)
ot_cost(::OptimalTransport.SqEuclidean, ::OptimalTransport.MvNormal, ::OptimalTransport.MvNormal)
wasserstein
squared2wasserstein
```

## Entropically regularised optimal transport

```@docs
sinkhorn
sinkhorn2
sinkhorn_barycenter
```

Currently the following variants of the Sinkhorn algorithm are supported:

```@docs
SinkhornGibbs
SinkhornStabilized
SinkhornEpsilonScaling
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
OptimalTransport.Dual.getprimal_ot_entropic_semidual
OptimalTransport.Dual.ot_entropic_dual
```
