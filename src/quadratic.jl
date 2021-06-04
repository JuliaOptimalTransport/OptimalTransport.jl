"""
    quadreg(mu, nu, C, ϵ; θ = 0.1, tol = 1e-5,maxiter = 50,κ = 0.5,δ = 1e-5)

Computes the optimal transport plan of histograms `mu` and `nu` with cost matrix `C` and quadratic regularization parameter `ϵ`,
using the semismooth Newton algorithm [Lorenz 2016].

This implementation makes use of IterativeSolvers.jl and SparseArrays.jl.

Parameters:\n
θ: starting Armijo parameter.\n
tol: tolerance of marginal error.\n
maxiter: maximum interation number.\n
κ: control parameter of Armijo.\n
δ: small constant for the numerical stability of conjugate gradient iterative solver.\n

Tips:
If the algorithm does not converge, try some different values of θ.

Reference:
Lorenz, D.A., Manns, P. and Meyer, C., 2019. Quadratically regularized optimal transport. arXiv preprint arXiv:1903.01112v4.
"""
function quadreg(mu, nu, C, ϵ; θ=0.1, tol=1e-5, maxiter=50, κ=0.5, δ=1e-5)
    if !(sum(mu) ≈ sum(nu))
        throw(ArgumentError("Error: mu and nu must lie in the simplex"))
    end

    N = length(mu)
    M = length(nu)

    # initialize dual potentials as uniforms
    a = ones(M) ./ M
    b = ones(N) ./ N
    γ = spzeros(M, N)

    da = spzeros(M)
    db = spzeros(N)

    converged = false

    function DualObjective(a, b)
        A = a .* ones(N)' + ones(M) .* b' - C'

        return 0.5 * norm(A[A .> 0], 2)^2 - ϵ * (dot(nu, a) + dot(mu, b))
    end

    # computes minimizing directions, update γ
    function search_dir!(a, b, da, db)
        P = a * ones(N)' .+ ones(M) * b' .- C'

        σ = 1.0 * sparse(P .>= 0)
        γ = sparse(max.(P, 0) ./ ϵ)

        G = vcat(
            hcat(spdiagm(0 => σ * ones(N)), σ),
            hcat(sparse(σ'), spdiagm(0 => sparse(σ') * ones(M))),
        )

        h = vcat(γ * ones(N) - nu, sparse(γ') * ones(M) - mu)

        x = cg(G + δ * I, -ϵ * h)

        da = x[1:M]
        return db = x[(M + 1):end]
    end

    function search_dir(a, b)
        P = a * ones(N)' .+ ones(M) * b' .- C'

        σ = 1.0 * sparse(P .>= 0)
        γ = sparse(max.(P, 0) ./ ϵ)

        G = vcat(
            hcat(spdiagm(0 => σ * ones(N)), σ),
            hcat(sparse(σ'), spdiagm(0 => sparse(σ') * ones(M))),
        )

        h = vcat(γ * ones(N) - nu, sparse(γ') * ones(M) - mu)

        x = cg(G + δ * I, -ϵ * h)

        return x[1:M], x[(M + 1):end]
    end

    # computes optimal maginitude in the minimizing directions
    function search_t(a, b, da, db, θ)
        d = ϵ * dot(γ, (da .* ones(N)' .+ ones(M) .* db')) - ϵ * (dot(da, nu) + dot(db, mu))

        ϕ₀ = DualObjective(a, b)
        t = 1

        while DualObjective(a + t * da, b + t * db) >= ϕ₀ + t * θ * d
            t *= κ

            if t < 1e-15
                # @warn "@ i = $i, t = $t , armijo did not converge"
                break
            end
        end
        return t
    end

    for i in 1:maxiter

        # search_dir!(a, b, da, db)
        da, db = search_dir(a, b)

        t = search_t(a, b, da, db, θ)

        a += t * da
        b += t * db

        err1 = norm(γ * ones(N) - nu, Inf)
        err2 = norm(sparse(γ') * ones(M) - mu, Inf)

        if err1 <= tol && err2 <= tol
            converged = true
            @warn "Converged @ i = $i with marginal errors: \n err1 = $err1, err2 = $err2 \n"
            break
        elseif i == maxiter
            @warn "Not Converged with errors:\n err1 = $err1, err2 = $err2 \n"
        end

        @debug " t = $t"
        @debug "marginal @ i = $i: err1 = $err1, err2 = $err2 "
    end

    if !converged
        @warn "SemiSmooth Newton algorithm did not converge"
    end

    return sparse(γ')
end
