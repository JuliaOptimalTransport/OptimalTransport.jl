using MathOptInterface

const MOI = MathOptInterface
"""
ToSimplexFormat(`\\mu,\\nu,C`)
Transforms the matrix format of the Discrete Optimal Transport
problem to the Linear Programming Format.

Variables `\\mu,\\nu` are the probability distributions.

```math
{\\bf A} = \\begin{bmatrix}
{\\bf 1_n}^T \\otimes \\mathbb I_m\\
\\mathbb I_n \\otimes {\\bf 1_m}^T
\\end{bmatrix} \\
b = [\\mu,\\nu]^T \\
c = [C_{1,1},C_{2,1},...,C_{n,1},C_{2,1},...,C_{n,m}]^T \\
p = [P_{1,1},P_{2,1},...,P_{n,1},P_{2,1},...,P_{n,m}]^T \\
```

The LP problem for OT is
```math
min c^T p \\
Ap = b \\
p \\geq 0\\
```
"""
function ToSimplexFormat(μ,ν,C)

	n,m = size(μ)[1],size(ν)[1]
	# If n=/= m, then fill with zeros until they are equal
	if size(μ)[1]!=size(ν)[1]
		if n - m > 0
		    d  = n-m
            C  = vcat(C,zeros(n,d))
		    ν  = vcat(ν,zeros(d));
		elseif n - m < 0
		    d = m-n
			C = hcat(C,zeros(m,d))
		    μ = vcat(μ,zeros(d))
		end
	end
	n,m = max(n,m),max(n,m); # They are now equal

	c   = reshape(C,n*m,1)[:] # Matrix to vector format
	b   = vcat(μ,ν)           # Distributions constraints
	A = [kron(ones(n)',1.0*Matrix(I,m,m))
     	 kron(1.0*Matrix(I,n,n),ones(m)')]

   	return c,A,b,μ,ν
end


"""
SolveLP(c,A,b)
Solve the following Linear Programming problem.
LP problem format:
       min  dot(c, x)
subject to A x = b
             0 ≤ x

'model' consists in the optimizer of choice by the user.
e.g: model = Tulip.Optimizer()
"""
function SolveLP(c,A,b,model::MOI.ModelLike)
	x = MOI.add_variables(model, length(c));
	MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
	            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, x), 0.0))
	MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
	for xi in x
	    MOI.add_constraint(model, MOI.SingleVariable(xi), MOI.GreaterThan(0.0))
	end

	for (i,row) in enumerate(eachrow(A))
	    row_function = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(row, x), 0.0);
	    MOI.add_constraint(model, row_function, MOI.EqualTo(b[i]))
	end
	MOI.optimize!(model)
	p = MOI.get(model, MOI.VariablePrimal(), x);
	return p
end

