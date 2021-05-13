# OptimalTransport.jl Documentation


## Exact optimal transport (Kantorovich) problem
```@docs
emd
emd2
```

### Exact optimal transport for 1D.

```@docs
otCost1D
otPlan1D
otCostPlan1D
```

For a cost function ``c(x,y) = h(|x-y|)`` where
The Optimal Transport Cost for 1D distributions can be obtained by:
```math
\\int_0^1 h(F_\\nu^{-1}(y) - F_\\mu^{-1}(y)) dy
```
Note that for the p-Wasserstein, we obtain:

```math
\\left(\\int_0^1 |F_\\nu^{-1}(y) - F_\\mu^{-1}(y)|^p dy\\right)^{1/p}
```

Where ``F_\\mu^{-1}`` is the quantile function for the distribution ``\\alpha``
(the inverse of the Cumulative Distribution Function).

## Entropically regularised optimal transport

```@docs
sinkhorn
sinkhorn2
sinkhorn_stabilized_epsscaling
sinkhorn_stabilized
sinkhorn_barycenter
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
