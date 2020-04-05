ENV["PYTHON"] = "/home/zsteve/anaconda3/bin/python"
Pkg.build("PyCall")
using PyCall
pot = pyimport("ot")

##
function emd(a, b, M)
    return pot.lp.emd(a, b, PyReverseDims(M))
end

function emd2(a, b, M)
    return pot.lp.emd2(a, b, PyReverseDims(M))
end
