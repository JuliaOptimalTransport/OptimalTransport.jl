using OptimalTransport

using CUDA
using Distances

using Test

@testset "gpu" begin
    # ensure that a GPU is available
    if CUDA.functional()
        @testset "Simple GPU" begin
            include("simple_gpu.jl")
        end
    else
        @warn "skipped GPU tests: no GPU available"
    end
end
