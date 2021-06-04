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

struct FiniteDiscreteMeasure{X<:AbstractArray,P<:AbstractVector}
    support::X
    p::P

    function FiniteDiscreteMeasure{X,P}(support::X, p::P) where {X,P}
        size(support, 1) == length(p) ||
            error("number of rows of `support` and `p` must be equal")
        sum(p) ≈ 1 || error("`p` must sum to 1")
        all(p .>= 0) || error("values of `p` must be greater of equal than 0")
        return new{X,P}(support, p)
    end
end

"""
    FiniteDiscreteMeasure(support::AbstractArray, p::AbstractVector)
Construct a finite discrete probability measure with support `support` and corresponding weights `p`.
"""
function FiniteDiscreteMeasure(support::AbstractArray, p::AbstractVector)
    P = sum(p)
    if size(support, 2) == 1
        return if P ≈ 1
            DiscreteNonParametric(vec(support), p)
        else
            DiscreteNonParametric(vec(support), p ./ P)
        end
    else
        return if P ≈ 1
            FiniteDiscreteMeasure{typeof(support),typeof(p)}(support, p)
        else
            FiniteDiscreteMeasure{typeof(support),typeof(p)}(support, p ./ P)
        end
    end
end

"""
    FiniteDiscreteMeasure(support::AbstractArray)
Construct a finite discrete probability measure with support `support` and equal probability for each point.
"""
function FiniteDiscreteMeasure(support::AbstractArray)
    p = ones(size(support)[1]) ./ size(support)[1]
    if size(support, 2) == 1
        return DiscreteNonParametric(vec(support), p)
    else
        return FiniteDiscreteMeasure{typeof(support),typeof(p)}(support, p)
    end
end

Distributions.support(d::FiniteDiscreteMeasure) = d.support
Distributions.probs(d::FiniteDiscreteMeasure) = d.p
