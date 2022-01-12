using OptimalTransport

μ = rand(10, 5)
ν = rand(10, 5)
C = rand(10, 10)

sinkhorn_unbalanced2(μ, ν, C, 1.0, 1.0, 1.0)

