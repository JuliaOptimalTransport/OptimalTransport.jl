using LinearAlgebra

"""
    bures_wasserstein_mapping_hd(ms, mt, Us, Ut, ls, lt, sigma_s2, sigma_t2)

Compute the optimal linear transport operator between two high-dimensional Gaussian distributions.

This function estimates the linear operator M that maps the source Gaussian distribution 
to the target Gaussian distribution, following the approach from Bouveyron & Corneli (2026), Theorem 2.10.

The linear operator has the form:
    M(x) = A * x + b

where A is a linear transformation matrix and b is a translation vector.

# Arguments
- `ms::AbstractVector`: Mean vector of the source distribution (size p)
- `mt::AbstractVector`: Mean vector of the target distribution (size p)
- `Us::AbstractMatrix`: Orthogonal matrix spanning the principal subspace of the source distribution (size p × ds)
- `Ut::AbstractMatrix`: Orthogonal matrix spanning the principal subspace of the target distribution (size p × dt)
- `ls::AbstractVector`: Variances along the principal axes for the source distribution (size ds)
- `lt::AbstractVector`: Variances along the principal axes for the target distribution (size dt)
- `sigma_s2::Real`: Residual variance of the source distribution (scalar)
- `sigma_t2::Real`: Residual variance of the target distribution (scalar)

# Returns
- `A::Matrix`: Linear transformation matrix (size p × p)
- `b::Vector`: Translation/bias vector (size p)


The optimal transport map is computed using a closed-form formula from Bouveyron & Corneli (2026), Theorem 2.10.

# References
- Knott, M. & Smith, C. S. (1984). "On the optimal mapping of distributions", 
  Journal of Optimization Theory and Applications, Vol 43.
  
- Bouveyron, C. & Corneli, M. (2026). "Scaling Optimal Transport to High-Dimensional 
  Gaussian Distributions".

# Examples


using LinearAlgebra

p = 100
ds = 10
dt = 15

ms = zeros(p)
mt = ones(p)

Us = Matrix(qr(randn(p, ds)).Q)
Ut = Matrix(qr(randn(p, dt)).Q)

ls = ones(ds)
lt = 2 .* ones(dt)

sigma_s2 = 0.1
sigma_t2 = 0.2

A, b = bures_wasserstein_mapping_hd(ms, mt, Us, Ut, ls, lt, sigma_s2, sigma_t2)

x = randn(p)
y = A * x + b
"""
function bures_wasserstein_mapping_hd(
    ms::AbstractVector,
    mt::AbstractVector,
    Us::AbstractMatrix,
    Ut::AbstractMatrix,
    ls::AbstractVector,
    lt::AbstractVector,
    sigma_s2::Real,
    sigma_t2::Real,
)
    sigma_s = sqrt(sigma_s2)

    Cs = Diagonal(sqrt.(ls .+ sigma_s2) .- sigma_s)
    Sigma_s_sqrt = Us * Cs * Us' + sigma_s * I

    Ds = Diagonal((sqrt.(ls .+ sigma_s2) .- sigma_s) ./ sqrt.(ls .+ sigma_s2))
    Sigma_s_invsqrt = (1 / sigma_s) * (I - Us * Ds * Us')

    Sigma_t = Ut * Diagonal(lt) * Ut' + sigma_t2 * I

    middle = sqrt(Symmetric(Sigma_s_sqrt * Sigma_t * Sigma_s_sqrt))

    A = Sigma_s_invsqrt * middle * Sigma_s_invsqrt
    b = mt - A * ms

    return A, b
end


"""
    empirical_bures_wasserstein_mapping_hd(xs, xt, d_intrinsic; reg=0.0, ws=nothing, wt=nothing, bias=true)

Compute the high-dimensional (HD) Bures-Wasserstein optimal transport linear mapping between empirical samples.

# Description

This function estimates the optimal linear HD operator that aligns two empirical distributions. 
It is equivalent to estimating the closed-form mapping between two HD Gaussian distributions.

The linear operator from source to target is defined as:

    M(x) = A * x + b

where:
- `A` is the linear operator matrix.
- `b` is the bias vector.

The matrices are computed using the following closed-form expressions:
- `A = Sigma_s^{-1/2} * (Sigma_s^{1/2} * Sigma_t * Sigma_s^{1/2})^{1/2} * Sigma_s^{-1/2}`
- `Sigma_s^{1/2} = sigma_s * I_p + U_s * C_s * U_s^T`
- `C_s = diag(sqrt(l_s + sigma_s^2) - sigma_s)`
- `Sigma_s^{-1/2} = (1 / sigma_s) * (I_p - U_s * D_s * U_s^T)`
- `D_s = diag((sqrt(l_s + sigma_s^2) - sigma_s) / sqrt(l_s + sigma_s^2))`
- `Sigma_t = U_t * diag(l_t) * U_t^T + sigma_t^2 * I_p`
- `b = mu_t - A * mu_s`

Assuming that the source and destination data samples have been generated from HD Gaussian 
distributions, the probabilistic PCA estimators are used to estimate the model parameters 
and plugged into the above formulas.

# Arguments

- `xs::AbstractMatrix`: Samples in the source domain (size: ns x p).
- `xt::AbstractMatrix`: Samples in the target domain (size: nt x p).
- `d_intrinsic::Union{Int, Tuple{Int, Int}}`: The intrinsic dimensions of the source and target 
  distributions. If an `Int` is provided, the same intrinsic dimension is assumed for both.

# Keyword Arguments

- `reg::Real=0.0`: Regularization added to the diagonals of covariances.
- `ws::Union{AbstractVector, Nothing}=nothing`: Weights for the source samples.
- `wt::Union{AbstractVector, Nothing}=nothing`: Weights for the target samples.
- `bias::Bool=true`: If `true`, estimates the bias `b`. If `false`, sets `b = 0`.

# Returns

- `A::Matrix`: The linear operator (size: p x p).
- `b::Vector`: The bias vector (size: p).

# References

- [1] Bouveyron, C. & Corneli, M. "Scaling Optimal Transport to High-Dimensional Gaussian Distributions".
- [2] Tipping, M.E. & Bishop, C.M. "Probabilistic Principal Component Analysis".
"""
function empirical_bures_wasserstein_mapping_hd(
    xs::AbstractMatrix,
    xt::AbstractMatrix,
    d_intrinsic::Union{Int,Tuple{Int,Int}};
    reg::Real=0.0,
    ws::Union{AbstractVector,Nothing}=nothing,
    wt::Union{AbstractVector,Nothing}=nothing,
    bias::Bool=true,
)
    ds, dt = d_intrinsic isa Integer ? (d_intrinsic, d_intrinsic) : d_intrinsic

    ns = size(xs, 1)
    nt = size(xt, 1)
    p = size(xs, 2)

    if isnothing(ws)
        ws = ones(eltype(xs), ns) ./ ns
    end

    if isnothing(wt)
        wt = ones(eltype(xt), nt) ./ nt
    end

    if bias
        mxs = vec(ws' * xs ./ sum(ws))
        mxt = vec(wt' * xt ./ sum(wt))

        xs = xs .- mxs'
        xt = xt .- mxt'
    else
        mxs = zeros(eltype(xs), p)
        mxt = zeros(eltype(xt), p)
    end

    Cs = (xs .* ws)' * xs ./ sum(ws) + reg * I
    Ct = (xt .* wt)' * xt ./ sum(wt) + reg * I

    eigs = eigen(Symmetric(Cs))
    vals = reverse(eigs.values)
    vecs = eigs.vectors[:, end:-1:1]

    a_s = vals[1:ds]
    sigma_2s = (tr(Cs) - sum(a_s)) / (p - ds)
    Us = vecs[:, 1:ds]
    ls = a_s .- sigma_2s

    eigt = eigen(Symmetric(Ct))
    valt = reverse(eigt.values)
    vect = eigt.vectors[:, end:-1:1]

    a_t = valt[1:dt]
    sigma_2t = (tr(Ct) - sum(a_t)) / (p - dt)
    Ut = vect[:, 1:dt]
    lt = a_t .- sigma_2t

    return bures_wasserstein_mapping_hd(mxs, mxt, Us, Ut, ls, lt, sigma_2s, sigma_2t)
end


"""
    bures_wasserstein_distance_hd(ms, mt, Us, Ut, ls, lt, sigma_s2, sigma_t2)

Compute the 2-Wasserstein distance between two high-dimensional Gaussian distributions.

This function implements the closed-form formula from Bouveyron & Corneli (2026), Proposition 2.3,
which allows computing the Wasserstein distance without explicitly forming the full covariance
matrices, making it scalable to high dimensions.

# Description

The two Gaussian distributions are represented in a low-rank + diagonal form:

    Source: N(ms, Sigma_s)  where  Sigma_s = Us * diag(ls) * Us' + sigma_s2 * I
    Target: N(mt, Sigma_t)  where  Sigma_t = Ut * diag(lt) * Ut' + sigma_t2 * I

The squared 2-Wasserstein distance is decomposed as:

    W2^2 = ||ms - mt||^2
         + tr(diag(ls))
         + tr(diag(lt))
         + p * (sigma_s2 + sigma_t2)
         - 2 * tr( sqrt( Sigma_s^(1/2) * Sigma_t * Sigma_s^(1/2) ) )

where Sigma_s^(1/2) is computed efficiently as:

    Sigma_s^(1/2) = sqrt(sigma_s2) * I + Us * Cs * Us'

and Cs is a diagonal matrix with entries:

    Cs[i] = sqrt(ls[i] + sigma_s2) - sqrt(sigma_s2)

# Arguments
- ms::AbstractVector: Mean vector of the source distribution (size p)
- mt::AbstractVector: Mean vector of the target distribution (size p)
- Us::AbstractMatrix: Orthogonal matrix spanning the principal subspace of the source (size p × ds)
- Ut::AbstractMatrix: Orthogonal matrix spanning the principal subspace of the target (size p × dt)
- ls::AbstractVector: Variances along the principal axes of the source (size ds)
- lt::AbstractVector: Variances along the principal axes of the target (size dt)
- sigma_s2::Real: Residual (isotropic) variance of the source distribution
- sigma_t2::Real: Residual (isotropic) variance of the target distribution

# Returns
- W::Real: The 2-Wasserstein distance between the two distributions

# References
- Knott, M. & Smith, C. S. (1984). "On the optimal mapping of distributions",
  Journal of Optimization Theory and Applications, Vol 43.
- Peyré, G. & Cuturi, M. (2019). "Computational Optimal Transport",
  Foundations and Trends in Machine Learning.
- Bouveyron, C. & Corneli, M. (2026). "Scaling Optimal Transport to
  High-Dimensional Gaussian Distributions".

# Example
p = 100  # ambient dimension
ds, dt = 10, 15  # intrinsic dimensions

ms = zeros(p)
Us = randn(p, ds)
ls = abs.(randn(ds))
sigma_s2 = 0.1

mt = ones(p)
Ut = randn(p, dt)
lt = abs.(randn(dt))
sigma_t2 = 0.2

W2 = bures_wasserstein_distance_hd(ms, mt, Us, Ut, ls, lt, sigma_s2, sigma_t2)
"""
function bures_wasserstein_distance_hd(
    ms::AbstractVector,
    mt::AbstractVector,
    Us::AbstractMatrix,
    Ut::AbstractMatrix,
    ls::AbstractVector,
    lt::AbstractVector,
    sigma_s2::Real,
    sigma_t2::Real,
)
    p = size(Us, 1)

    sigma_s = sqrt(sigma_s2)

    Cs = Diagonal(sqrt.(ls .+ sigma_s2) .- sigma_s)
    Sigma_s_sqrt = Us * Cs * Us' + sigma_s * I

    Sigma_t = Ut * Diagonal(lt) * Ut' + sigma_t2 * I

    middle = Sigma_s_sqrt * Sigma_t * Sigma_s_sqrt

    W2 = (
        sum(abs2, ms .- mt)
        + sum(ls)
        + sum(lt)
        + p * (sigma_s2 + sigma_t2)
        - 2 * tr(sqrt(Symmetric(middle)))
    )

    return sqrt(max(real(W2), zero(real(W2))))
end




"""
    empirical_bures_wasserstein_distance_hd(xs, xt; d_intrinsic, reg=0.0, ws=nothing, wt=nothing, bias=true)

Estimate the  2-Wasserstein distance between two high-dimensional Gaussian distributions from raw data samples.

This function first estimates the parameters of the source and target Gaussian distributions 
(means, principal subspaces, and variances) using Probabilistic PCA (PPCA), and then plugs 
these estimates into the closed-form Bures-Wasserstein distance formula.

# Description

The squared 2-Wasserstein distance between the estimated distributions is computed as:

    W2^2 = ||ms - mt||^2 
         + tr(Ls) + tr(Lt) 
         + p * (sigma_s2 + sigma_t2) 
         - 2 * tr( sqrt( Sigma_s^(1/2) * Sigma_t * Sigma_s^(1/2) ) )

where the covariance matrices are estimated via PPCA as:
    Sigma_s = Us * diag(ls) * Us' + sigma_s2 * I
    Sigma_t = Ut * diag(lt) * Ut' + sigma_t2 * I

# Arguments
- xs::AbstractMatrix: Samples from the source distribution (size ns x p)
- xt::AbstractMatrix: Samples from the target distribution (size nt x p)
- d_intrinsic::Union{Integer, Tuple{Integer, Integer}}`: The intrinsic dimension(s) of the distributions. 
  If an integer is provided, the same intrinsic dimension is used for both source and target. 
  If a tuple is provided, it specifies (ds, dt) respectively.
- reg::Real: Regularization parameter added to the diagonal of the estimated covariances to ensure 
  positive definiteness (default: 0.0).
- ws::Union{AbstractVector, Nothing}: Optional weights for the source samples (default: nothing, uniform weights).
- wt::Union{AbstractVector, Nothing}: Optional weights for the target samples (default: nothing, uniform weights).
- bias::Bool: If true, estimates the means (ms, mt) from the data. If false, assumes the means are zero (default: true).

# Returns
- W::Real: The estimated 2-Wasserstein distance between the two distributions.

# References
- Bouveyron, C. & Corneli, M. (2024). "Scaling Optimal Transport to High-Dimensional Gaussian Distributions".
- Tipping, M.E. & Bishop, C.M. (1999). "Probabilistic Principal Component Analysis", 
  Journal of the Royal Statistical Society: Series B.

# Example

p = 100      # ambient dimension
ns, nt = 500, 600  # number of samples
ds, dt = 10, 15    # intrinsic dimensions

# Generate dummy data
xs = randn(ns, p)
xt = randn(nt, p)

# Compute empirical distance
W2 = empirical_bures_wasserstein_distance_hd(xs, xt; d_intrinsic=(ds, dt), reg=1e-4)
"""
function empirical_bures_wasserstein_distance_hd(
    xs::AbstractMatrix,
    xt::AbstractMatrix;
    d_intrinsic::Union{Integer,Tuple{<:Integer,<:Integer}},
    reg::Real=0.0,
    ws::Union{AbstractVector,Nothing}=nothing,
    wt::Union{AbstractVector,Nothing}=nothing,
    bias::Bool=true,
)

    ds, dt = d_intrinsic isa Integer ? (d_intrinsic, d_intrinsic) : d_intrinsic

    ns = size(xs, 1)
    nt = size(xt, 1)
    p = size(xs, 2)

    if isnothing(ws)
        ws = ones(eltype(xs), ns) ./ ns
    end

    if isnothing(wt)
        wt = ones(eltype(xt), nt) ./ nt
    end

    if bias
        mxs = vec(ws' * xs ./ sum(ws))
        mxt = vec(wt' * xt ./ sum(wt))

        xs = xs .- mxs'
        xt = xt .- mxt'
    else
        mxs = zeros(eltype(xs), p)
        mxt = zeros(eltype(xt), p)
    end

    Cs = (xs .* ws)' * xs ./ sum(ws) + reg * I
    Ct = (xt .* wt)' * xt ./ sum(wt) + reg * I

    eigs = eigen(Symmetric(Cs))
    vals = reverse(eigs.values)
    vecs = eigs.vectors[:, end:-1:1]

    a_s = vals[1:ds]
    sigma_2s = (tr(Cs) - sum(a_s)) / (p - ds)
    Us = vecs[:, 1:ds]
    ls = a_s .- sigma_2s

    eigt = eigen(Symmetric(Ct))
    valt = reverse(eigt.values)
    vect = eigt.vectors[:, end:-1:1]

    a_t = valt[1:dt]
    sigma_2t = (tr(Ct) - sum(a_t)) / (p - dt)
    Ut = vect[:, 1:dt]
    lt = a_t .- sigma_2t

    return bures_wasserstein_distance_hd(mxs, mxt, Us, Ut, ls, lt, sigma_2s, sigma_2t)
end
        
    
            
    