using JuMP
using Tulip
using Clp

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
	extra_bfs   = [] # store indexes where zeros were added
	if size(μ)[1]!=size(ν)[1]
		if n - m > 0
		    d  = n-m
            C  = vcat(C,zeros(n,d))
		    ν  = vcat(ν,zeros(d));
		    for i = -1:d-2
		    	# extra_bfs is used on the Simplex
		        extra_bfs = vcat(extra_bfs,n + n*(n-1 - 1 - i))
		    end
		elseif n - m < 0
		    d = m-n
			C = hcat(C,zeros(m,d))
		    μ = vcat(μ,zeros(d))
		    for i = -1:d-2
		    	# extra_bfs is used on the Simplex
		        extra_bfs = vcat(extra_bfs,m-1 - i + m*(m - 1))
		    end
		end
	end
	n,m = max(n,m),max(n,m); # They are now equal

	c   = reshape(C,n*m,1)[:] # Matrix to vector format
	b   = vcat(μ,ν)           # Distributions constraints
	A = [kron(ones(n)',1.0*Matrix(I,m,m))
     	 kron(1.0*Matrix(I,n,n),ones(m)')]

   	return c,A,b,μ,ν,extra_bfs
end



"""
NorthWest_Rule(a,b)
Return a Basic Feasible Solution to the Earth-Mover
Optimal Transport Problem,
to be used as initial point for Simplex algorithm.

`a` and `b` are the marginal distributions, hence, they must be arrays.
return `P`, a n by m matrix.
"""
function NorthWest_Rule(a,b)
    n,m = size(a)[1],size(b)[1]
    P = zeros(n,m)
    i,j = 1,1
    r,c = a[1],b[1]
    while i <= n && j <= m
        t = min(r,c)
        P[i,j] = t
        r = r - t
        c = c - t
        if r==0
            i = i + 1
            if i<= n
                r = a[i]
            end
        end
        if c == 0
            j = j + 1
            if j<= m
                c = b[j]
            end
        end
    end
    return P
  end

"""
WARNING - This implementation is slow! It was done with educational
purposes so anyone can see how the Simplex is to be implemented.
SimplexFromBFS(c,A,b, initial_bfs; max_iterations=100, index_bfs=[0])
Starting from a basic feasible point, uses the Simplex
algorithm to minimize the LP problem in the format:
       min  dot(c, x)
subject to A x = b
             0 ≤ x

The function returns `x^*` representing the argument
that minimizes the LP problem.

`index_bfs` is an optional parameter which consists
of manually providing the active restrictions.
It's use becomes necessary when the vertex used for
initializing the algorithm contains values equal
to zero, hence, the algorithm is unable to decide which
restrictions are active.
"""
function SimplexNative(c,A,b,
        initial_bfs;max_iterations=100,index_bfs=[0])
    
    # Initial setup
    e  = 10^-7
    c = -c
    if index_bfs[1] == 0
        B  = findall(initial_bfs .> 0+e)
        N  = findall(initial_bfs .<= 0+e)
    else
        B = index_bfs
        N = []
        for i in collect(1:size(initial_bfs)[1])
            if !(i in B)
               push!(N,i) 
            end
        end
    end
    xn = initial_bfs[N]; xb = initial_bfs[B];
    p  = 0
    # Simplex pivoting iteration
    for i = 1:max_iterations
        Ab = A[:,B]; An = A[:,N]; cb = c[B]; cn = c[N]
        p  = inv(Ab)*b
        Q  = -inv(Ab)*An
        r  = (cb'*Q + cn')'
        if all(r.<= 0+e)
            x_final = vcat(hcat(B,p),hcat(N,zeros(length(N))))
            x_final = x_final[sortperm(x_final[:,1]),:]
            return x_final[:,2]
        end
        zo = cb'*p
#         z  = zo + r'*xn
        index_in =findmax(r)[2]
        x_in = N[index_in]
        if any(Q[:,index_in] .< 0)
            coef_entering = -p./Q[:,index_in] 
            q_neg_index   = findall(Q[:,index_in] .< 0)
            index_out     = findfirst(coef_entering .== findmin(coef_entering[q_neg_index])[1])
            x_out         = B[index_out]
            B[index_out]  = x_in
            N[index_in]   = x_out
        else
            
            error("Unbounded")
        end
        if i == max_iterations
        	println("Max Iterations reached.")
        end
    end
    x_final = vcat(hcat(B,p),hcat(N,zeros(length(N))))
    x_final = x_final[sortperm(x_final[:,1]),:]
    return x_final[:,2]
end

"""
SimplexClp(c,A,b)
Simplex algorithm using Clp Optimizer and JuMP for modeling.

LP problem format:
       min  dot(c, x)
subject to A x = b
             0 ≤ x
"""
function SimplexClp(c,A,b)
	model = JuMP.Model(Clp.Optimizer)
	@variable(model, x[1:size(c)[1]])
	@constraint(model,A*x.==b)
	@constraint(model,x.>=0)
	@objective(model, Min, c'*x)
	optimize!(model)
	p = zeros(size(c)[1])
	for i = 1:size(c)[1]
	    p[i] = value(x[i])
	end
	return p
end

"""
InteriorPointMethod(c,A,b)
Interior Point Method using Tulip Optimizer and JuMP for modeling.

LP problem format:
       min  dot(c, x)
subject to A x = b
             0 ≤ x
WARNING: If any of the marginal distributions has only one mass
point (e.g.: mu = [0.25,0.25,0.5], nu = [1]), then the function
will return an error, since the LP problem will not have an interior.
"""
function InteriorPointMethod(c,A,b)
	model = JuMP.Model(Tulip.Optimizer)
	@variable(model, x[1:size(c)[1]])
	@constraint(model,A*x.==b)
	@constraint(model,x.>=0)
	@objective(model, Min, c'*x)
	optimize!(model)
	p = zeros(size(c)[1])
	for i = 1:size(c)[1]
	    p[i] = value(x[i])
	end
	return p
end
