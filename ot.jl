# ENV["PYTHON"] = "/home/zsteve/anaconda3/bin/python"
Pkg.build("PyCall")
using PyCall
pot = pyimport("ot")

# using JuMP
# using Clp
# using Distances

# function emd(mu, nu, C, optim = with_optimizer(Clp.Optimizer, LogLevel = 0))
#     # return pot.lp.emd(a, b, PyReverseDims(M))'
#     m = Model(optim)
#     @variable(m, 0 <= x[1:length(mu), 1:length(nu)] <= 1)
#     @objective(m, Min, sum(x.*C))
#     @constraint(m, x*ones(length(nu)) .== mu)
#     @constraint(m, x'*ones(length(mu)) .== nu)
#     optimize!(m)
#     return value.(x)
# end
#
# function emd2(mu, nu, C)
#     return sum(emd(mu, nu, C).*C)
# end

function emd(a, b, M)
    return pot.lp.emd(b, a, PyReverseDims(M))'
end

function emd2(a, b, M)
    return pot.lp.emd2(b, a, PyReverseDims(M))
end

## test
# N = 50
# mu_spt = rand(N)
# mu = rand(N)
# mu = mu/sum(mu)
# 
# nu_spt = rand(2*N)
# nu = rand(2*N)
# nu = nu/sum(nu)
# 
# 
# 
# C = (pairwise(Euclidean(), mu_spt', nu_spt')).^2
# 
# γ = emd(mu, nu, C)
# sum(γ.*C)
# 
# emd2(mu, nu, C)
