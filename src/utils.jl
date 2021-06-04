"""
     add_singleton(x::AbstractArray, ::Val{dim}) where {dim}

Add an additional dimension `dim` of size 1 to array `x`.
"""
function add_singleton(x::AbstractArray, ::Val{dim}) where {dim}
    shape = ntuple(max(ndims(x) + 1, dim)) do i
        return i < dim ? size(x, i) : (i > dim ? size(x, i - 1) : 1)
    end
    return reshape(x, shape)
end

"""
    dot_matwise(x::AbstractArray, y::AbstractArray)

Compute the inner product of all matrices in `x` and `y`.

At least one of `x` and `y` has to be a matrix.
"""
dot_matwise(x::AbstractMatrix, y::AbstractMatrix) = dot(x, y)
function dot_matwise(x::AbstractArray, y::AbstractMatrix)
    xmat = reshape(x, size(x, 1) * size(x, 2), :)
    return reshape(reshape(y, 1, :) * xmat, size(x)[3:end])
end
dot_matwise(x::AbstractMatrix, y::AbstractArray) = dot_matwise(y, x)

"""
    checksize2(x::AbstractVecOrMat, y::AbstractVecOrMat)

Check if arrays `x` and `y` are compatible, then return a tuple of its broadcasted second
dimension.
"""
checksize2(::AbstractVector, ::AbstractVector) = ()
function checksize2(μ::AbstractVecOrMat, ν::AbstractVecOrMat)
    size_μ_2 = size(μ, 2)
    size_ν_2 = size(ν, 2)
    if size_μ_2 > 1 && size_ν_2 > 1 && size_μ_2 != size_ν_2
        throw(DimensionMismatch("size of source and target marginals is not compatible"))
    end
    return (max(size_μ_2, size_ν_2),)
end

"""
     checkbalanced(μ::AbstractVecOrMat, ν::AbstractVecOrMat)

Check that source and target marginals `μ` and `ν` are balanced.
"""
function checkbalanced(μ::AbstractVector, ν::AbstractVector)
    sum(μ) ≈ sum(ν) || throw(ArgumentError("source and target marginals are not balanced"))
    return nothing
end
function checkbalanced(x::AbstractVecOrMat, y::AbstractVecOrMat)
    all(isapprox.(sum(x; dims=1), sum(y; dims=1))) ||
        throw(ArgumentError("source and target marginals are not balanced"))
    return nothing
end

"""
    cost_matrix(
        c,
        μ::Union{FiniteDiscreteMeasure, DiscreteNonParametric},
        ν::Union{FiniteDiscreteMeasure, DiscreteNonParametric}
    )

Compute cost matrix from Finite Discrete Measures `μ` and `ν` using cost function  `c`.

Note that the use of functions such as `SqEuclidean()` from `Distances.jl` have
better performance than generic functions. Thus, it's prefered to use
`cost_matrix(SqEuclidean(), μ, ν)`, instead of `cost_matrix((x,y)->sum((x-y).^2), μ, ν)`
or even `cost_matrix(sqeuclidean, μ, ν)`.

For custom cost functions, it is necessary to guarantee that the function `c` works
on vectors, i.e., if one wants to compute the squared Euclidean distance,
the one must define `c(x,y) = sum((x - y).^2)`.

# Example
```julia
μ = FiniteDiscreteMeasure(rand(10),rand(10))
ν = FiniteDiscreteMeasure(rand(8))
c = TotalVariation()
C = cost_matrix(c, μ, ν)
```
"""
function cost_matrix(
    c,
    μ::Union{FiniteDiscreteMeasure, DiscreteNonParametric},
    ν::Union{FiniteDiscreteMeasure, DiscreteNonParametric}
)
    if typeof(c) <: PreMetric && size(μ.support,2) == 1
        return pairwise(c, μ.support, ν.support)
    elseif typeof(c) <: PreMetric && size(μ.support,2) > 1
        return pairwise(c, μ.support, ν.support, dims=1)
    else
        return pairwise(c, eachrow(μ.support), eachrow(ν.support))
    end
end

"""
    cost_matrix(
        c,
        μ::Union{FiniteDiscreteMeasure, DiscreteNonParametric},
        symmetric = false
    )

Compute cost matrix from Finite Discrete Measures `μ` to itself using cost function  `c`.
If the cost function is symmetric, set the argument `symmetric` to `true` in order
to increase performance.
"""
function cost_matrix(
    c,
    μ::Union{FiniteDiscreteMeasure, DiscreteNonParametric};
    symmetric = false
)
    if typeof(c) <: PreMetric && size(μ.support,2) == 1
        return pairwise(c, μ.support)
    elseif typeof(c) <: PreMetric && size(μ.support,2) > 1
        return pairwise(c, μ.support, dims=1)
    else
        return pairwise(c, eachrow(μ.support), symmetric=symmetric)
    end
end