"""
    wasserstein(μ, ν; metric=Euclidean(), p=Val(1), kwargs...)

Compute the `p`-Wasserstein distance with respect to the `metric` between measures `μ` and
`ν`.

Order `p` can be provided as a scalar of type `Real` or as a parameter of a value type
`Val(p)`. For certain combinations of `metric` and `p`, such as `metric=Euclidean()` and
`p=Val(2)`, the computations are more efficient if `p` is specified as a value type. The
remaining keyword arguments are forwarded to [`ot_cost`](@ref).

See also: [`squared2wasserstein`](@ref), [`ot_cost`](@ref)
"""
function wasserstein(μ, ν; metric=Euclidean(), p::Union{Real,Val}=Val(1), kwargs...)
    cost = ot_cost(p2distance(metric, p), μ, ν; kwargs...)
    return prt(cost, p)
end

# compute the cost function corresponding to a metric and exponent `p`
p2distance(metric, ::Val{1}) = metric
p2distance(metric, ::Val{P}) where {P} = (x, y) -> metric(x, y)^P
p2distance(d::Euclidean, ::Val{2}) = SqEuclidean(d.thresh)
p2distance(metric, p) = (x, y) -> metric(x, y)^p

# compute the `p` root
prt(x, ::Val{1}) = x
prt(x, ::Val{2}) = sqrt(x)
prt(x, ::Val{3}) = cbrt(x)
prt(x, ::Val{P}) where {P} = x^(1 / P)
prt(x, p) = x^(1 / p)

"""
    squared2wasserstein(μ, ν; metric=Euclidean(), kwargs...)

Compute the squared 2-Wasserstein distance with respect to the `metric` between measures `μ`
and `ν`.

The remaining keyword arguments are forwarded to [`ot_cost`](@ref).

See also: [`wasserstein`](@ref), [`ot_cost`](@ref)
"""
function squared2wasserstein(μ, ν; metric=Euclidean(), kwargs...)
    return ot_cost(p2distance(metric, Val(2)), μ, ν; kwargs...)
end
