<<<<<<< HEAD
using Documenter, OptimalTransport

makedocs(sitename="OptimalTransport.jl")
=======
using OptimalTransport

using Documenter
using PyCall

makedocs(;
    modules=[OptimalTransport, POT],
    repo="https://github.com/zsteve/OptimalTransport.jl/blob/{commit}{path}#L{line}",
    sitename="OptimalTransport.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://zsteve.github.io/OptimalTransport.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "POT" => "pot.md",
    ],
)

deploydocs(;
    repo="github.com/zsteve/OptimalTransport.jl",
)
>>>>>>> upstream/master
