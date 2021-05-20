"""
optimal_transport_cost(c,mu::ContinuousUnivariateDistribution,nu::UnivariateDistribution; plan=nothing)

Calculates the optimal transport cost between μ to ν, where
they are 1-Dimensional distributions and the cost
function is of the form ``c(x,y) = h(|x-y|)`` such that
``h`` is a convex function.

Parameters:\n
c: The cost function, which should be of the form ``c(x,y) = h(abs(x-y))``
where `h` is a convex function.
mu: 1-D distribution (e.g. `mu = Distributions.Normal(0,1)`)\n
nu: 1-D distribution (e.g. `nu = Distributions.Normal(0,1)`)\n
plan: Optional parameter in case the optimal transport plan is already known. Providing
it can speed up calculations.
"""
function optimal_transport_cost(
    c, μ::ContinuousUnivariateDistribution, ν::UnivariateDistribution; plan=nothing
)
    if plan === nothing
        g(μ, ν, x) = c(quantile(μ, x), quantile(ν, x))
        f(x) = g(μ, ν, x)
        return quadgk(f, 0, 1)[1]
    else
        quadgk(0, 1) do q
            x = quantile(μ, q)
        return c(x, plan(x))
    end
    end
end

"""
optimal_transport_plan(c,mu::ContinuousUnivariateDistribution,nu::UnivariateDistribution)

Calculates the pptimal transport plan between μ to ν, where
they are 1-Dimensional distributions and the cost
function is of the form ``c(x,y) = h(|x-y|)`` such that
``h`` is a convex function.

Parameters:\n
c: The cost function, which should be of the form ``c(x,y) = h(|x-y|)``
where ``h`` is a convex function.
μ: 1-D distribution (e.g. `μ = Distributions.Normal(0,1)`)\n
ν: 1-D distribution (e.g. `μ = Distributions.Normal(0,1)`)\n

Returns the optimal transport plan as a function
  ```math
  T(x)=F_\\nu^{-1}(F_\\mu(x)).
  ```
"""
function optimal_transport_plan(
    c, μ::ContinuousUnivariateDistribution, ν::UnivariateDistribution
)
    # Use T instead of γ to indicate that this is a Monge map.
    T(x) = Distributions.quantile(ν, Distributions.cdf(μ, x))
    return T
end

"""
optimal_transport_cost(c, μ::DiscreteNonParametric, ν::DiscreteNonParametric; plan=nothing)

Calculates the Optimal Transport Cost between μ to ν, where
they are 1-Dimensional distributions and the cost
function is of the form ``c(x,y) = h(|x-y|)`` such that
``h`` is a convex function.

Parameters:\n
c: The cost function, which should be of the form ``c(x,y) = h(abs(x-y))``
where ``h`` is a convex function. \n
μ: Finite Discrete Distribution (e.g. `μ = DiscreteNonParametric(u, u_n)`)\n
ν: Finite Discrete Distribution (e.g. `ν = DiscreteNonParametric(v, v_n)`)\n

Returns the cost.
"""
function optimal_transport_cost(
    c, μ::DiscreteNonParametric, ν::DiscreteNonParametric; plan=nothing
)
    if plan === nothing
        return _optimal_transport_cost_plan(c, μ, ν)[1]
    else
        return dot(StatsBase.pairwise(c,μ.support,ν.support), plan)
    end

    return cost
end

"""
optimal_transport_plan(c, μ::DiscreteNonParametric, ν::DiscreteNonParametric; plan=nothing)

Calculates the Optimal Transport Plan between μ to ν, where
they are 1-Dimensional distributions and the cost
function is of the form ``c(x,y) = h(|x-y|)`` such that
``h`` is a convex function.

Parameters:\n
c: The cost function, which should be of the form ``c(x,y) = h(abs(x-y))``
where ``h`` is a convex function. \n
μ: Finite Discrete Distribution (e.g. `μ = DiscreteNonParametric(u, u_n)`)\n
ν: Finite Discrete Distribution (e.g. `ν = DiscreteNonParametric(v, v_n)`)\n

Returns the optimal transport plan γ as a matrix.
"""
function optimal_transport_plan(
    c, μ::DiscreteNonParametric, ν::DiscreteNonParametric; plan=nothing
)
    return _optimal_transport_cost_plan(c,μ,ν)[2]
end

"""
function _optimal_transport_cost_plan(
    c, μ::DiscreteNonParametric, ν::DiscreteNonParametric
)

Auxiliary function that calculates the cost and plan for
μ an ν where both  they are 1-Dimensional discrete distributions with finite support
and the cost function is of the form ``c(x,y) = h(|x-y|)`` such that
``h`` is a convex function.

Returns cost and γ, where cost represtents the optimal transport cost and
γ is the optimal transport plan given as a matrix.
"""
function _optimal_transport_cost_plan(
    c, μ::DiscreteNonParametric, ν::DiscreteNonParametric
)
    cost  = 0
    len_μ = length(μ.p)
    len_ν = length(ν.p)
    γ = zeros(len_μ, len_ν)

    wi = μ.p[1]
    wj = ν.p[1]
    i, j = 1, 1
    while true
        if (wi < wj || j == len_ν)
            γ[i, j] = wi
            cost += c(μ.support[i], ν.support[j]) * wi
            i += 1
            if i == len_μ + 1
                break
            end
            wj -= wi
            wi = μ.p[i]
        else
            γ[i, j] = wj
            cost += c(μ.support[i], ν.support[j]) * wj
            j += 1
            if j == len_ν + 1
                break
            end
            wi -= wj
            wj = ν.p[j]
        end
    end

    return cost, γ
end