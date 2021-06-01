# Code from @devmotion
# https://github.com/devmotion/\
# CalibrationErrorsDistributions.jl/blob/main/src/distances/bures.jl

"""
    tr_sqrt(A::AbstractMatrix)
Compute `tr(sqrt(A))`.
"""
tr_sqrt(A::AbstractMatrix) = LinearAlgebra.tr(sqrt(A))
tr_sqrt(A::PDMats.PDMat) = tr_sqrt(A.mat)
tr_sqrt(A::PDMats.PDiagMat) = sum(sqrt, A.diag)
tr_sqrt(A::PDMats.ScalMat) = A.dim * sqrt(A.value)

"""
    _gaussian_ot_A(A::AbstractMatrix, B::AbstractMatrix)
Compute
```math
sqrt(sqrt(A) B sqrt(A))
```
"""
function _gaussian_ot_A(A::AbstractMatrix, B::AbstractMatrix)
    sqrt_A = sqrt(A)
    return sqrt_A * B * sqrt_A
end
function _gaussian_ot_A(A::PDMats.PDiagMat, B::AbstractMatrix)
    return sqrt.(A.diag) .* B .* sqrt.(A.diag')
end
function _gaussian_ot_A(A::StridedMatrix, B::PDMats.PDMat)
    return PDMats.X_A_Xt(B, sqrt(A))
end
_gaussian_ot_A(A::PDMats.PDMat, B::PDMats.PDMat) = _gaussian_ot_A(A.mat, B)
_gaussian_ot_A(A::AbstractMatrix, B::PDMats.PDiagMat) = _gaussian_ot_A(B, A)
_gaussian_ot_A(A::PDMats.PDMat, B::StridedMatrix) = _gaussian_ot_A(B, A)

# """
#     _sqbures_kernel(A::AbstractMatrix, B::AbstractMatrix)
# Compute
# ```math
# tr(sqrt(sqrt(A) B sqrt(A)))
# ```
# """
# function _sqbures_kernel(A::AbstractMatrix, B::AbstractMatrix)
#     sqrt_A = sqrt(A)
#     return tr_sqrt(sqrt_A * B * sqrt_A)
# end
# function _sqbures_kernel(A::PDMats.PDiagMat, B::AbstractMatrix)
#     return tr_sqrt(sqrt.(A.diag) .* B .* sqrt.(A.diag'))
# end
# function _sqbures_kernel(A::StridedMatrix, B::PDMats.PDMat)
#     return tr_sqrt(PDMats.X_A_Xt(B, sqrt(A)))
# end
# _sqbures_kernel(A::PDMats.PDMat, B::PDMats.PDMat) = _sqbures_kernel(A.mat, B)
# _sqbures_kernel(A::AbstractMatrix, B::PDMats.PDiagMat) = _sqbures_kernel(B, A)
# _sqbures_kernel(A::PDMats.PDMat, B::StridedMatrix) = _sqbures_kernel(B, A)

"""
    sqbures(A::AbstractMatrix, B::AbstractMatrix)
Compute the squared Bures metric
```math
tr(A) + tr(B) - 2 tr(sqrt(sqrt(A) B sqrt(A))).
```
"""
function sqbures(A::AbstractMatrix, B::AbstractMatrix)
    # return LinearAlgebra.tr(A) + LinearAlgebra.tr(B) - 2 * _sqbures_kernel(A, B)
    return LinearAlgebra.tr(A) + LinearAlgebra.tr(B) - 2 * tr_sqrt(_gaussian_ot_A(A, B))
end

# diagonal matrix
function sqbures(A::PDMats.PDiagMat, B::PDMats.PDiagMat)
    return sum(zip(A.diag, B.diag)) do (x, y)
        abs2(sqrt(x) - sqrt(y))
    end
end

# scaled identity matrix
function sqbures(A::PDMats.ScalMat, B::AbstractMatrix)
    return LinearAlgebra.tr(A) + LinearAlgebra.tr(B) - 2 * sqrt(A.value) * tr_sqrt(B)
end
sqbures(A::AbstractMatrix, B::PDMats.ScalMat) = sqbures(B, A)
sqbures(A::PDMats.ScalMat, B::PDMats.ScalMat) = A.dim * abs2(sqrt(A.value) - sqrt(B.value))

# combinations
function sqbures(A::PDMats.PDiagMat, B::PDMats.ScalMat)
    sqrt_B = sqrt(B.value)
    return sum(A.diag) do x
        abs2(sqrt(x) - sqrt_B)
    end
end
sqbures(A::PDMats.ScalMat, B::PDMats.PDiagMat) = sqbures(B, A)
