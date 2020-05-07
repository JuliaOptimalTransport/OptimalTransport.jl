# OptimalTransport.jl Documentation


## Exact optimal transport (Kantorovich) problem
```@docs
emd(a, b, M)
emd2(a, b, M)
```

## Entropically regularised optimal transport

```@docs
sinkhorn_impl
sinkhorn
sinkhorn2
sinkhorn_stabilized_epsscaling
sinkhorn_stabilized
```

Wrapper functions to POT library:
```@docs
_sinkhorn
_sinkhorn2
_sinkhorn_stabilized_epsscaling
_sinkhorn_stabilized_epsscaling2
```

## Unbalanced optimal transport
```@docs
sinkhorn_unbalanced
sinkhorn_unbalanced2
```

