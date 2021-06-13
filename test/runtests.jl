using LinearAlgebra: symmetric
using OptimalTransport
using Pkg: Pkg
using SafeTestsets

using Test

const GROUP = get(ENV, "GROUP", "All")

@testset "OptimalTransport" begin
    if GROUP == "All" || GROUP == "OptimalTransport"
        @safetestset "Utilities" begin
            include("utils.jl")
        end
        @safetestset "Exact OT" begin
            include("exact.jl")
        end
        @testset "Entropically regularized OT" begin
            @safetestset "Sinkhorn" begin
                include(joinpath("entropic", "sinkhorn.jl"))
            end
            @safetestset "Stabilized Sinkhorn" begin
                include(joinpath("entropic", "sinkhorn_stabilized.jl"))
            end
        end
        @safetestset "Quadratically regularized OT" begin
            include("quadratic.jl")
        end
        @safetestset "Unbalanced OT" begin
            include("unbalanced.jl")
        end
        @safetestset "Wasserstein distance" begin
            include("wasserstein.jl")
        end
        @safetestset "Bures distance" begin
            include("bures.jl")
        end
    end

    # CUDA requires Julia >= 1.6
    if (GROUP == "All" || GROUP == "GPU") && VERSION >= v"1.6"
        # activate separate environment: CUDA can't be added to test/Project.toml since it
        # is not available on older Julia versions
        pkgdir = dirname(dirname(pathof(OptimalTransport)))
        Pkg.activate("gpu")
        Pkg.develop(Pkg.PackageSpec(; path=pkgdir))
        Pkg.instantiate()

        @safetestset "Simple GPU" begin
            include(joinpath("gpu/simple_gpu.jl"))
        end
    end
end
