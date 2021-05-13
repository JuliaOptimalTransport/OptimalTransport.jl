"""
otCost1d(c,μ::Distributions.UnivariateDistribution,
           ν::Distributions.UnivariateDistribution)

Calculates the Optimal Transport Cost between μ to ν, where
they are 1-Dimensional distributions and the cost
function is of the form ``c(x,y) = h(|x-y|)`` such that
``h`` is a convex function.

Parameters:\n
c: The cost function, which should be of the form ``c(x,y) = h(abs(x-y))``
where `h` is a convex function.
μ: 1-D distribution (e.g. `μ = Distributions.Normal(0,1)`)\n
ν: 1-D distribution (e.g. `μ = Distributions.Normal(0,1)`)\n
"""
function otCost1d(c,μ::Distributions.UnivariateDistribution,ν::Distributions.UnivariateDistribution)
    g(μ,ν,x) = c(quantile(μ,x),quantile(ν,x))
    f(x) = g(μ,ν,x)
    return quadgk(f,0,1)[1]
end

"""
otPlan1d(c,μ::Distributions.UnivariateDistribution,
            ν::Distributions.UnivariateDistribution)

Calculates the Optimal Transport Plan between μ to ν, where
they are 1-Dimensional distributions and the cost
function is of the form ``c(x,y) = h(|x-y|)`` such that
``h`` is a convex function.

Parameters:\n
c: The cost function, which should be of the form ``c(x,y) = h(|x-y|)``
where ``h`` is a convex function.
μ: 1-D distribution (e.g. `μ = Distributions.Normal(0,1)`)\n
ν: 1-D distribution (e.g. `μ = Distributions.Normal(0,1)`)\n

Returns the Optimal Transport Plan as a function
  ```math
  T(x)=F_\\nu^{-1}(F_\\mu(x)).
  ```
"""
function otPlan1d(c,μ::Distributions.UnivariateDistribution,ν::Distributions.UnivariateDistribution)
  T(x) = Distributions.quantile(ν,Distributions.cdf(μ,x))
  return T
end


"""
otCost1d(c,μ::Distributions.UnivariateDistribution,
            ν::Distributions.UnivariateDistribution)`

Calculates the Optimal Transport Cost between μ to ν, where
they are 1-Dimensional distributions and the cost
function is of the form ``c(x,y) = h(|x-y|)`` such that
``h`` is a convex function.

Parameters:\n
c: The cost function, which should be of the form ``c(x,y) = h(abs(x-y))``
where ``h`` is a convex function.
μ: 1-D distribution (e.g. `μ = Distributions.Normal(0,1)`)\n
ν: 1-D distribution (e.g. `μ = Distributions.Normal(0,1)`)\n

Returns the Optimal Transport Cost and the Plan as a function
  ```math
  T(x)=F_\\nu^{-1}(F_\\mu(x)).
  ```
"""
function otCostPlan1d(c,μ::Distributions.UnivariateDistribution,ν::Distributions.UnivariateDistribution)
    g(μ,ν,x) = c(quantile(μ,x),quantile(ν,x))
    f(x) = g(μ,ν,x)
    T(x) = Distributions.quantile(ν,Distributions.cdf(μ,x))
    return quadgk(f,0,1)[1], T
end

# """
# otCost1d(c,u::Vector,u_weights::Vector,v::Vector,v_weights::Vector)

# Calculates the Optimal Transport Cost between μ to ν, where
# they are 1-Dimensional distributions and the cost
# function is of the form ``c(x,y) = h(|x-y|)`` such that
# ``h`` is a convex function.

# Parameters:\n
# c: The cost function, which should be of the form ``c(x,y) = h(abs(x-y))``
# where ``h`` is a convex function.\n
# μ: 1-D distribution (e.g. `μ = Distributions.Normal(0,1)`)\n
# ν: 1-D distribution (e.g. `μ = Distributions.Normal(0,1)`)\n
# """

# function otCost1d(c,u::Vector,u_weights::Vector,v::Vector,v_weights::Vector)

#   @assert length(u) == length(u_weights)
#   @assert length(v) == length(v_weights)
#   @assert sum(u_weights) ≈ 1.0  atol=1e-7
#   @assert sum(v_weights) ≈ 1.0  atol=1e-7

#   cost = 0
#   perm_u = sortperm(u)
#   perm_v = sortperm(v)
#   u_sorted = u[perm_u]
#   v_sorted = v[perm_v]
#   u_w_sorted = u_weights[perm_u]
#   v_w_sorted = v_weights[perm_v]
  
#   wi = u_w_sorted[1]
#   wj = v_w_sorted[1]
#   i,j = 1,1
#   while true
#     if (wi < wj || j == length(v_w_sorted))
#       cost += c(u_sorted[i],v_sorted[j])*wi
#       i+=1
#       if i == length(u_sorted)+1
#           break
#       end
#       wj -= wi
#       wi  = u_w_sorted[i]
#     else
#       cost += c(u_sorted[i],v_sorted[j])*wj
#       j+=1
#       if j == length(v_sorted)+1
#           break
#       end
#       wi -= wj
#       wj  = v_w_sorted[j]
#     end
#   end
    
#   return cost
# end

# """
# otPlan1d(c,u::Vector,u_weights::Vector,v::Vector,v_weights::Vector)

# Calculates the Optimal Transport Plan between μ to ν, where
# they are 1-Dimensional distributions and the cost
# function is of the form ``c(x,y) = h(|x-y|)`` such that
# ``h`` is a convex function.

# Parameters:\n
# c: The cost function, which should be of the form ``c(x,y) = h(abs(x-y))``
# where ``h`` is a convex function.\n
# μ: 1-D distribution (e.g. `μ = Distributions.Normal(0,1)`)\n
# ν: 1-D distribution (e.g. `μ = Distributions.Normal(0,1)`)\n

# Returns the Optimal Transport Plan γ as a matrix.
# """

# function otPlan1d(c,u::Vector,u_weights::Vector,v::Vector,v_weights::Vector)
#   γ = zeros(length(u),length(v))
#   perm_u = sortperm(u)
#   perm_v = sortperm(v)
#   u_sorted = u[perm_u]
#   v_sorted = v[perm_v]
#   u_w_sorted = u_weights[perm_u]
#   v_w_sorted = v_weights[perm_v]
  
#   wi = u_w_sorted[1]
#   wj = v_w_sorted[1]
#   i,j = 1,1
#   while true
#       if (wi < wj || j == length(v_w_sorted))
#           γ[perm_u[i],perm_v[j]] = wi
#           i+=1
#           if i == length(u_sorted)+1
#               break
#           end
#           wj -= wi
#           wi  = u_w_sorted[i]
#       else
#           γ[perm_u[i],perm_v[j]] = wj
#           j+=1
#           if j == length(v_sorted)+1
#               break
#           end
#           wi -= wj
#           wj  = v_w_sorted[j]
#       end
#   end
  
#   return γ
# end

# """
# otPlan1d(c,u::Vector,u_weights::Vector,v::Vector,v_weights::Vector)

# Calculates the Optimal Transport Plan between μ to ν, where
# they are 1-Dimensional distributions and the cost
# function is of the form ``c(x,y) = h(|x-y|)`` such that
# ``h`` is a convex function.

# Parameters:\n
# c: The cost function, which should be of the form ``c(x,y) = h(abs(x-y))``
# where ``h`` is a convex function.\n
# μ: 1-D distribution (e.g. `μ = Distributions.Normal(0,1)`)\n
# ν: 1-D distribution (e.g. `μ = Distributions.Normal(0,1)`)\n

# Returns the Optimal Transport Plan γ as a matrix.
# """
# function otCostPlan1d(c,u::Vector,u_weights::Vector,v::Vector,v_weights::Vector)
#   γ = zeros(length(u),length(v))
#   cost = 0
#   perm_u = sortperm(u)
#   perm_v = sortperm(v)
#   u_sorted = u[perm_u]
#   v_sorted = v[perm_v]
#   u_w_sorted = u_w[perm_u]
#   v_w_sorted = v_w[perm_v]
  
#   wi = u_w_sorted[1]
#   wj = v_w_sorted[1]
#   i,j = 1,1
#   while true
#       if (wi < wj || j == length(v_w_sorted))
#           cost += c(u_sorted[i],v_sorted[j])*wi
#           γ[perm_u[i],perm_v[j]] = wi
#           i+=1
#           if i == length(u_sorted)+1
#               break
#           end
#           wj -= wi
#           wi  = u_w_sorted[i]
#       else
#           γ[perm_u[i],perm_v[j]] = wj
#           cost += c(u_sorted[i],v_sorted[j])*wj
#           j+=1
#           if j == length(v_sorted)+1
#               break
#           end
#           wi -= wj
#           wj  = v_w_sorted[j]
#       end
#   end
  
#   return cost, γ
# end


# function otCost1d(c,μ::Distributions.DiscreteNonParametric
#                     ν::Distributions.DiscreteNonParametric)
#   return otCost1d(c,μ.support,μ.p,ν.support,ν.p)
# end

# function otPlan1d(c,μ::Distributions.DiscreteNonParametric
#                     ν::Distributions.DiscreteNonParametric)
#   return otPlan1d(c,μ.support,μ.p,ν.support,ν.p)
# end

# """
# μ: Finite discrete distribution (i.e. ``μ := \sum^n_{i=0}w_i \\delta_{u_i})
# returns γ as a matrix. ,
# """
# function otCostPlan1d(c,μ::Distributions.DiscreteNonParametric
#                     ν::Distributions.DiscreteNonParametric)
#   return otCostPlan1d(c,μ.support,μ.p,ν.support,ν.p)
# end