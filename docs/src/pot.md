# Wrapper functions for the POT library

If you load the `PyCall` package, you get access to wrappers of functions
in the [Python Optimal Transport (POT) package](https://pythonot.github.io/).
The wrapper functions are available in the submodule `POT`.

## Exact optimal transport (Kantorovich) problem

```@docs
POT.emd
POT.emd2
```

## Entropically regularised optimal transport

```@docs
POT.sinkhorn
POT.sinkhorn2
```

## Unbalanced optimal transport

```@docs
POT.sinkhorn_unbalanced
POT.sinkhorn_unbalanced2
```