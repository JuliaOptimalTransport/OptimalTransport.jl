"""
    toSimplexFormat(μ, ν, C)

Formulate the discrete optimal transport problem with source `μ`, target `ν`, and
cost matrix `C` as a linear programming problem.

More concretely, this function returns a tuple of the variables `c`, `A`, and `b` of the
LP problem
```math
\\begin{aligned}
\\min_{p} c^T p & \\\\
\\text{subject to } Ap &= b \\\\
0 &\\leq p
\\end{aligned}
```
where
```math
\\begin{aligned}
p &= [P_{1,1},P_{2,1},\\ldots,P_{n,1},P_{2,1},\\ldots,P_{n,m}]^T, \\\\
c &= [C_{1,1},C_{2,1},\\ldots,C_{n,1},C_{2,1},\\ldots,C_{n,m}]^T, \\\\
A &= \\begin{bmatrix}
1_n^T \\otimes I_m \\\\
I_n \\otimes 1_m^T
\\end{bmatrix}, \\\\
b &= [μ, ν]^T.
\\end{aligned}
```
"""
function toSimplexFormat(μ, ν, C)
    m = length(μ)
    n = length(ν)

    c = vec(C)
    b = Vcat(μ, ν)
    A = Vcat(
        Kron(Ones(1, n), Eye(m)),
        Kron(Eye(n), Ones(1, m)),
    )

    return c, A, b
end

"""
    solveLP(c, A, b, optimizer)

Solve the linear programming problem
```math
\\begin{aligned}
\\min_{x} c^T x & \\\\
\\text{subject to } Ax &= b \\\\
0 &\\leq x
\\end{aligned}
```
with the given `optimizer`.

Possible choices of `optimizer` are `Tulip.Optimizer()` or `Clp.Optimizer()` in the
`Tulip` and `Clp` packages, respectively.
"""
function solveLP(c, A, b, model::MOI.ModelLike)
    x = MOI.add_variables(model, length(c))
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{eltype(c)}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, x), zero(eltype(c))),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    for xi in x
        MOI.add_constraint(model, MOI.SingleVariable(xi), MOI.GreaterThan(0.0))
    end

    for (row, bi) in zip(eachrow(A), b)
        row_function = MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.(row, x), zero(eltype(row)),
        )
        MOI.add_constraint(model, row_function, MOI.EqualTo(bi))
    end

    MOI.optimize!(model)
    p = MOI.get(model, MOI.VariablePrimal(), x)

    return p
end
