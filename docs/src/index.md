# OptimalTransport.jl Documentation


## Exact optimal transport (Kantorovich) problem

```@docs
emd
emd2
ot_plan
ot_plan(::Any, ::ContinuousUnivariateDistribution, ::UnivariateDistribution)
ot_plan(::Any, ::DiscreteNonParametric, ::DiscreteNonParametric)
ot_cost
ot_cost(::Any, ::ContinuousUnivariateDistribution, ::UnivariateDistribution)
ot_cost(::Any, ::DiscreteNonParametric, ::DiscreteNonParametric)
wasserstein
squared2wasserstein
```

## Entropically regularised optimal transport

```@docs
sinkhorn
sinkhorn2
sinkhorn_stabilized_epsscaling
sinkhorn_stabilized
sinkhorn_barycenter
```

Currently the following variants of the Sinkhorn algorithm are supported:

```@docs
SinkhornGibbs
SinkhornStabilized
SinkhornEpsilonScaling
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
