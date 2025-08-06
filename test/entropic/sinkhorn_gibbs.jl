using OptimalTransport

using Distances
using ForwardDiff
using ReverseDiff
using LogExpFunctions
using PythonOT: PythonOT

using LinearAlgebra
using Random
using Test

const POT = PythonOT

Random.seed!(100)

@testset "sinkhorn_gibbs.jl" begin
    # size of source and target
    M = 250
    N = 200

    # create two random histograms
    μ = normalize!(rand(M), 1)
    ν = normalize!(rand(N), 1)

    # create random cost matrix
    C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

    # regularization parameter
    ε = 0.1

    @testset "example" begin
        # compute optimal transport plan and optimal transport cost
        γ = sinkhorn(μ, ν, C, ε, SinkhornGibbs())
        c = sinkhorn2(μ, ν, C, ε, SinkhornGibbs())

        # check that plan and cost are consistent
        @test c ≈ dot(γ, C)

        # compare with default algorithm
        γ_default = sinkhorn(μ, ν, C, ε)
        c_default = sinkhorn2(μ, ν, C, ε)
        @test γ_default == γ
        @test c_default == c

        # compare with POT
        γ_pot = POT.sinkhorn(μ, ν, C, ε; stopThr=1e-16)
        c_pot = POT.sinkhorn2(μ, ν, C, ε; stopThr=1e-16)[1]
        @test γ_pot ≈ γ rtol = 1e-6
        @test c_pot ≈ c rtol = 1e-7

        # compute optimal transport cost with regularization term
        c_w_regularization = sinkhorn2(μ, ν, C, ε, SinkhornGibbs(); regularization=true)
        @test c_w_regularization ≈ c + ε * sum(x -> iszero(x) ? x : x * log(x), γ)
        @test c_w_regularization == sinkhorn2(μ, ν, C, ε; regularization=true)

        # ensure that provided plan is used and correct
        c2 = sinkhorn2(similar(μ), similar(ν), C, rand(), SinkhornGibbs(); plan=γ)
        @test c2 ≈ c
        @test c2 == sinkhorn2(similar(μ), similar(ν), C, rand(); plan=γ)
        c2_w_regularization = sinkhorn2(
            similar(μ), similar(ν), C, ε, SinkhornGibbs(); plan=γ, regularization=true
        )
        @test c2_w_regularization ≈ c_w_regularization
        @test c2_w_regularization ==
            sinkhorn2(similar(μ), similar(ν), C, ε; plan=γ, regularization=true)

        # batches of histograms
        d = 10
        for (size2_μ, size2_ν) in
            (((), (d,)), ((1,), (d,)), ((d,), ()), ((d,), (1,)), ((d,), (d,)))
            # generate batches of histograms
            μ_batch = repeat(μ, 1, size2_μ...)
            ν_batch = repeat(ν, 1, size2_ν...)

            # compute optimal transport plan and check that it is consistent with the
            # plan for individual histograms
            γ_all = sinkhorn(μ_batch, ν_batch, C, ε, SinkhornGibbs())
            @test size(γ_all) == (M, N, d)
            @test all(view(γ_all, :, :, i) ≈ γ for i in axes(γ_all, 3))
            @test γ_all == sinkhorn(μ_batch, ν_batch, C, ε)

            # compute optimal transport cost and check that it is consistent with the
            # cost for individual histograms
            c_all = sinkhorn2(μ_batch, ν_batch, C, ε, SinkhornGibbs())
            @test size(c_all) == (d,)
            @test all(x ≈ c for x in c_all)
            @test c_all == sinkhorn2(μ_batch, ν_batch, C, ε)
        end
    end

    # different element type
    @testset "Float32" begin
        # create histograms and cost matrix with element type `Float32`
        μ32 = map(Float32, μ)
        ν32 = map(Float32, ν)
        C32 = map(Float32, C)
        ε32 = Float32(ε)

        # compute optimal transport plan and optimal transport cost
        γ = sinkhorn(μ32, ν32, C32, ε32, SinkhornGibbs())
        c = sinkhorn2(μ32, ν32, C32, ε32, SinkhornGibbs())
        @test eltype(γ) === Float32
        @test typeof(c) === Float32

        # check that plan and cost are consistent
        @test c ≈ dot(γ, C32)

        # compare with default algorithm
        γ_default = sinkhorn(μ32, ν32, C32, ε32)
        c_default = sinkhorn2(μ32, ν32, C32, ε32)
        @test γ_default == γ
        @test c_default == c

        # compare with POT
        γ_pot = POT.sinkhorn(μ32, ν32, C32, ε32)
        c_pot = POT.sinkhorn2(μ32, ν32, C32, ε32)[1]
        @test map(Float32, γ_pot) ≈ γ rtol = 1e-3
        @test Float32(c_pot) ≈ c rtol = 1e-3

        # batches of histograms
        d = 10
        for (size2_μ, size2_ν) in
            (((), (d,)), ((1,), (d,)), ((d,), ()), ((d,), (1,)), ((d,), (d,)))
            # generate batches of histograms
            μ32_batch = repeat(μ32, 1, size2_μ...)
            ν32_batch = repeat(ν32, 1, size2_ν...)

            # compute optimal transport plan and check that it is consistent with the
            # plan for individual histograms
            γ_all = sinkhorn(μ32_batch, ν32_batch, C32, ε32, SinkhornGibbs())
            @test size(γ_all) == (M, N, d)
            @test all(view(γ_all, :, :, i) ≈ γ for i in axes(γ_all, 3))
            @test γ_all == sinkhorn(μ32_batch, ν32_batch, C32, ε32)

            # compute optimal transport cost and check that it is consistent with the
            # cost for individual histograms
            c_all = sinkhorn2(μ32_batch, ν32_batch, C32, ε32, SinkhornGibbs())
            @test size(c_all) == (d,)
            @test all(x ≈ c for x in c_all)
            @test c_all == sinkhorn2(μ32_batch, ν32_batch, C32, ε32)
        end
    end

    # https://github.com/JuliaOptimalTransport/OptimalTransport.jl/issues/86
    @testset "AD" begin
        # compute gradients with respect to source and target marginals separately and
        # together. test against gradient computed using analytic formula of Proposition 2.3 of 
        # Cuturi, Marco, and Gabriel Peyré. "A smoothed dual approach for variational Wasserstein problems." SIAM Journal on Imaging Sciences 9.1 (2016): 320-343.
        #
        ε = 0.1 # use a larger ε to avoid having to do many iterations 
        # target marginal
        for Diff in [ReverseDiff, ForwardDiff]
            ∇ = Diff.gradient(log.(ν)) do xs
                sinkhorn2(μ, softmax(xs), C, ε, SinkhornGibbs(); regularization=true)
            end
            ∇default = Diff.gradient(log.(ν)) do xs
                sinkhorn2(μ, softmax(xs), C, ε; regularization=true)
            end
            @test ∇ == ∇default

            solver = OptimalTransport.build_solver(μ, ν, C, ε, SinkhornGibbs())
            OptimalTransport.solve!(solver)
            # helper function
            function dualvar_to_grad(x, ε)
                x = -ε * log.(x)
                x .-= sum(x) / size(x, 1)
                return -x
            end
            ∇_ot = dualvar_to_grad(solver.cache.v, ε)
            # chain rule because target measure parameterised by softmax
            J_softmax = ForwardDiff.jacobian(log.(ν)) do xs
                softmax(xs)
            end
            ∇analytic_target = J_softmax * ∇_ot
            # check that gradient obtained by AD matches the analytic formula
            @test ∇ ≈ ∇analytic_target

            # source marginal
            ∇ = Diff.gradient(log.(μ)) do xs
                sinkhorn2(softmax(xs), ν, C, ε, SinkhornGibbs(); regularization=true)
            end
            ∇default = Diff.gradient(log.(μ)) do xs
                sinkhorn2(softmax(xs), ν, C, ε; regularization=true)
            end
            @test ∇ == ∇default

            # check that gradient obtained by AD matches the analytic formula
            solver = OptimalTransport.build_solver(μ, ν, C, ε, SinkhornGibbs())
            OptimalTransport.solve!(solver)
            J_softmax = ForwardDiff.jacobian(log.(μ)) do xs
                softmax(xs)
            end
            ∇_ot = dualvar_to_grad(solver.cache.u, ε)
            ∇analytic_source = J_softmax * ∇_ot
            @test ∇ ≈ ∇analytic_source

            # both marginals
            ∇ = Diff.gradient(log.(vcat(μ, ν))) do xs
                sinkhorn2(
                    softmax(xs[1:M]),
                    softmax(xs[(M + 1):end]),
                    C,
                    ε,
                    SinkhornGibbs();
                    regularization=true,
                )
            end
            ∇default = Diff.gradient(log.(vcat(μ, ν))) do xs
                sinkhorn2(softmax(xs[1:M]), softmax(xs[(M + 1):end]), C, ε; regularization=true)
            end
            @test ∇ == ∇default
            ∇analytic = vcat(∇analytic_source, ∇analytic_target)
            @test ∇ ≈ ∇analytic
        end
    end

    @testset "deprecations" begin
        # check `sinkhorn`
        γ = sinkhorn(μ, ν, C, ε; atol=1e-6)
        @test (@test_deprecated sinkhorn(μ, ν, C, ε; tol=1e-6)) == γ
        γ = sinkhorn(μ, ν, C, ε; check_convergence=5)
        @test (@test_deprecated sinkhorn(μ, ν, C, ε; check_marginal_step=5)) == γ

        # check `sinkhorn2`
        c = sinkhorn2(μ, ν, C, ε; atol=1e-6)
        @test (@test_deprecated sinkhorn2(μ, ν, C, ε; tol=1e-6)) == c
        c = sinkhorn2(μ, ν, C, ε; check_convergence=5)
        @test (@test_deprecated sinkhorn2(μ, ν, C, ε; check_marginal_step=5)) == c
    end
end
