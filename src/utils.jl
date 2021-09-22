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

function dot_vecwise(x::AbstractMatrix, y::AbstractMatrix)
    return [
        dot(u, v) for (u, v) in
        zip((view(x, :, i) for i in axes(x, 2)), (view(y, :, i) for i in axes(y, 2)))
    ]
end

dot_vecwise(x::AbstractMatrix, y::AbstractVector) = x' * y

dot_vecwise(x::AbstractVector, y::AbstractMatrix) = dot_vecwise(y, x)
dot_vecwise(x::AbstractVector, y::AbstractVector) = dot(x, y)

"""
    checksize(μ::AbstractVecOrMat, ν::AbstractVecOrMat, C::AbstractMatrix)

Check that dimensions of source and target marginals `μ` and `ν` are consistent with cost
matrix `C`.
"""
function checksize(μ::AbstractVecOrMat, ν::AbstractVecOrMat, C::AbstractMatrix)
    size(C) == (size(μ, 1), size(ν, 1)) || throw(
        DimensionMismatch("cost matrix `C` must be of size `(size(μ, 1), size(ν, 1))`")
    )
    return nothing
end

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
     isallapprox(x::AbstractVector)

Check that all entries of `x` are approximately equal
"""
function isallapprox(x::AbstractVecOrMat)
    return all(y -> isapprox(y, x[1]), x[2:end])
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
function checkbalanced(x::AbstractMatrix)
    isallapprox(sum(x; dims=1)) ||
        throw(ArgumentError("source and target marginals are not balanced"))
    return nothing
end

"""
    A_batched_mul_B!(c::AbstractVector, A::AbstractMatrix, b::AbstractVector)

Compute the matrix-vector product `Ab` and write the result to `c`.
"""
A_batched_mul_B!(c::AbstractVector, A::AbstractMatrix, b::AbstractVector) = mul!(c, A, b)

"""
    A_batched_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)

Compute the matrix-matrix product `AB` and write the result to `C`.
"""
A_batched_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = mul!(C, A, B)

"""
    A_batched_mul_B!(C::AbstractMatrix, A::AbstractArray{<:Any,3}, B::AbstractMatrix)

Compute the matrix-vector products `A[:, :, i] B[:, i]` and write them to `C[:, i]`.
"""
function A_batched_mul_B!(C::AbstractMatrix, A::AbstractArray{<:Any,3}, B::AbstractMatrix)
    return NNlib.batched_mul!(add_singleton(C, Val(2)), A, add_singleton(B, Val(2)))
end

"""
    At_batched_mul_B!(c::AbstractVector, A::AbstractMatrix, b::AbstractVector)

Compute the matrix-vector product `transpose(A)b` and write the result to `c`.
"""
function At_batched_mul_B!(c::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    return mul!(c, transpose(A), b)
end

"""
    At_batched_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)

Compute the matrix-matrix product `transpose(A)B` and write the result to `C`.
"""
function At_batched_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    return mul!(C, transpose(A), B)
end

"""
    At_batched_mul_B!(C::AbstractMatrix, A::AbstractArray{<:Any,3}, B::AbstractMatrix)

Compute the matrix-vector products `transpose(A[:, :, i]) B[:, i]` and write them to
`C[:, i]`.
"""
function At_batched_mul_B!(C::AbstractMatrix, A::AbstractArray{<:Any,3}, B::AbstractMatrix)
    return NNlib.batched_mul!(add_singleton(C, Val(1)), add_singleton(B, Val(1)), A)
end
