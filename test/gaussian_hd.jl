using Test
using LinearAlgebra
using Random
using Statistics

# Import only the four new functions added to OptimalTransport.jl.
using OptimalTransport:
    bures_wasserstein_distance_hd,
    empirical_bures_wasserstein_distance_hd,
    bures_wasserstein_mapping_hd,
    empirical_bures_wasserstein_mapping_hd

# Build a p x d matrix with orthonormal columns.
# The HD Gaussian formulas assume that U spans an orthonormal principal subspace.
function orthonormal_columns(p, d)
    return Matrix(qr(randn(p, d)).Q)
end

# Reconstruct the full dense covariance matrix from the HD Gaussian model:
#
#     Sigma = U * Diagonal(l) * U' + sigma2 * I
#
# This is used only in tests to compare the HD implementation with a dense
# reference formula in small dimension.
function covariance_hd(U, l, sigma2)
    p = size(U, 1)
    return U * Diagonal(l) * U' + sigma2 * Matrix{Float64}(I, p, p)
end

# Dense reference formula for the Bures-Wasserstein distance between two
# Gaussian distributions N(ms, Cs) and N(mt, Ct).
#
# This is not the scalable HD formula. It explicitly uses dense covariance
# matrices, so it is suitable as a correctness check for small dimensions.
function dense_bures_wasserstein_distance(ms, mt, Cs, Ct)
    Cs_sqrt = sqrt(Symmetric(Cs))
    middle = Cs_sqrt * Ct * Cs_sqrt

    W2 = (
        sum(abs2, ms .- mt)
        + tr(Cs)
        + tr(Ct)
        - 2 * tr(sqrt(Symmetric(middle)))
    )

    # The max protects against tiny negative values caused by roundoff,
    # for example W2 = -1e-14 instead of exactly zero.
    return sqrt(max(real(W2), 0.0))
end

# Dense reference formula for the optimal affine transport map:
#
#     T(x) = A * x + b
#
# between two Gaussian distributions N(ms, Cs) and N(mt, Ct).
#
# We use it to check that the HD implementation returns the same A and b
# as the classical dense formula.
function dense_bures_wasserstein_mapping(ms, mt, Cs, Ct)
    Cs_sqrt = sqrt(Symmetric(Cs))
    Cs_invsqrt = inv(Cs_sqrt)

    middle = sqrt(Symmetric(Cs_sqrt * Ct * Cs_sqrt))

    A = Cs_invsqrt * middle * Cs_invsqrt
    b = mt - A * ms

    return A, b
end

@testset "HD Gaussian Bures-Wasserstein distance" begin
    Random.seed!(1234)

    # Use a small ambient dimension so that the dense reference computation
    # is cheap and numerically stable.
    p = 20
    ds = 4
    dt = 5

    # The means are different, so the test also checks the ||ms - mt|| term.
    ms = zeros(p)
    mt = vcat(3.0, zeros(p - 1))

    # Principal subspaces for the source and target HD Gaussian models.
    Us = orthonormal_columns(p, ds)
    Ut = orthonormal_columns(p, dt)

    # Principal variances in each low-dimensional subspace.
    ls = [6.0, 4.0, 2.0, 1.0]
    lt = [5.0, 3.0, 2.0, 1.0, 0.5]

    # Residual isotropic variances.
    sigma_s2 = 0.7
    sigma_t2 = 0.4

    # Reconstruct full covariance matrices to compute a dense reference result.
    Cs = covariance_hd(Us, ls, sigma_s2)
    Ct = covariance_hd(Ut, lt, sigma_t2)

    # Distance computed by the new HD implementation.
    W_hd = bures_wasserstein_distance_hd(ms, mt, Us, Ut, ls, lt, sigma_s2, sigma_t2)

    # Distance computed by the classical dense formula.
    W_dense = dense_bures_wasserstein_distance(ms, mt, Cs, Ct)

    # Main correctness check: the HD formula should match the dense formula.
    @test W_hd ≈ W_dense rtol = 1e-8 atol = 1e-8

    # Basic numerical sanity checks.
    @test isfinite(W_hd)
    @test W_hd >= 0

    # Identity case: the distance between a Gaussian and itself should be zero.
    W_same = bures_wasserstein_distance_hd(ms, ms, Us, Us, ls, ls, sigma_s2, sigma_s2)

    @test W_same ≈ 0 atol = 1e-8
end

@testset "HD Gaussian Bures-Wasserstein mapping" begin
    Random.seed!(5678)

    p = 20
    ds = 4
    dt = 5

    ms = zeros(p)
    mt = vcat(2.0, -1.0, zeros(p - 2))

    Us = orthonormal_columns(p, ds)
    Ut = orthonormal_columns(p, dt)

    ls = [6.0, 4.0, 2.0, 1.0]
    lt = [5.0, 3.0, 2.0, 1.0, 0.5]

    sigma_s2 = 0.7
    sigma_t2 = 0.4

    Cs = covariance_hd(Us, ls, sigma_s2)
    Ct = covariance_hd(Ut, lt, sigma_t2)

    # Transport map computed by the new HD implementation.
    A_hd, b_hd = bures_wasserstein_mapping_hd(ms, mt, Us, Ut, ls, lt, sigma_s2, sigma_t2)

    # Transport map computed by the dense reference formula.
    A_dense, b_dense = dense_bures_wasserstein_mapping(ms, mt, Cs, Ct)

    # Main correctness checks: A and b should match the dense reference.
    @test A_hd ≈ A_dense rtol = 1e-8 atol = 1e-8
    @test b_hd ≈ b_dense rtol = 1e-8 atol = 1e-8

    # Basic numerical sanity checks.
    @test all(isfinite, A_hd)
    @test all(isfinite, b_hd)

    # Simple case: same covariance, different means.
    # The optimal map should be a pure translation:
    #
    #     T(x) = x + (mt - ms)
    #
    # Therefore A should be I and b should be mt - ms.
    A_same, b_same = bures_wasserstein_mapping_hd(ms, mt, Us, Us, ls, ls, sigma_s2, sigma_s2)

    @test A_same ≈ Matrix{Float64}(I, p, p) atol = 1e-7
    @test b_same ≈ mt - ms atol = 1e-7
end

@testset "Empirical HD Gaussian Bures-Wasserstein distance" begin
    Random.seed!(42)

    n = 300
    p = 15
    d = 3

    # Generate a source sample cloud.
    xs = randn(n, p)

    # The target sample cloud is exactly the source cloud translated by shift.
    # Therefore the empirical covariances are identical and only the means differ.
    shift = collect(range(0.1, 1.5; length=p))
    xt = xs .+ shift'

    # Since only the mean changes, the expected Wasserstein distance is norm(shift).
    W = empirical_bures_wasserstein_distance_hd(xs, xt; d_intrinsic=d, reg=1e-8, bias=true)

    @test W ≈ norm(shift) atol = 1e-6
    @test isfinite(W)
    @test W >= 0

    # Identity case: same source and target samples should give zero distance.
    W_same = empirical_bures_wasserstein_distance_hd(xs, xs; d_intrinsic=d, reg=1e-8, bias=true)

    @test W_same <= 1e-7
end

@testset "Empirical HD Gaussian Bures-Wasserstein mapping" begin
    Random.seed!(43)

    n = 300
    p = 15
    d = 3

    xs = randn(n, p)

    # The target is a translated copy of the source.
    # The expected optimal map is therefore A = I and b = shift.
    shift = collect(range(-0.5, 1.0; length=p))
    xt = xs .+ shift'

    A, b = empirical_bures_wasserstein_mapping_hd(xs, xt, d; reg=1e-8, bias=true)

    # Check output dimensions.
    @test size(A) == (p, p)
    @test size(b) == (p,)

    # Check that the output does not contain NaN or Inf values.
    @test all(isfinite, A)
    @test all(isfinite, b)

    # Because source and target have the same empirical covariance, up to
    # translation, the optimal map should be A = I and b = shift.
    @test A ≈ Matrix{Float64}(I, p, p) atol = 1e-6
    @test b ≈ shift atol = 1e-6

    # Samples are stored by rows: xs has shape n x p.
    # The mathematical formula is T(x) = A * x + b for column vectors.
    # For row-wise samples, the equivalent operation is xs * A' .+ b'.
    Xst = xs * A' .+ b'

    # After applying the transport map, the transported sample mean should
    # match the target sample mean.
    @test vec(mean(Xst; dims=1)) ≈ vec(mean(xt; dims=1)) atol = 1e-6

    # Empirical identity case: source and target are exactly the same sample cloud.
    # The expected map is A = I and b = 0.
    A_same, b_same = empirical_bures_wasserstein_mapping_hd(xs, xs, d; reg=1e-8, bias=true)

    @test A_same ≈ Matrix{Float64}(I, p, p) atol = 1e-6
    @test b_same ≈ zeros(p) atol = 1e-6
end