# ENV["PYTHON"] = "/home/zsteve/anaconda3/bin/python"
Pkg.build("PyCall")
using PyCall
pot = pyimport("ot")

function emd(a, b, M)
    return pot.lp.emd(b, a, PyReverseDims(M))'
end

function emd2(a, b, M)
    return pot.lp.emd2(b, a, PyReverseDims(M))
end

function sinkhorn(a, b, M, eps)
    return pot.sinkhorn(b, a, PyReverseDims(M), eps)'
end

function sinkhorn2(a, b, M, eps)
    return pot.sinkhorn2(b, a, PyReverseDims(M), eps)[1]
end
