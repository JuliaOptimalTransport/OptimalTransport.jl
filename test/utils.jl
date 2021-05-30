using OptimalTransport

using LinearAlgebra

@testset "utils.jl" begin
    @testset "add_singleton" begin
        x = rand(3)
        y = @inferred(OptimalTransport.add_singleton(x, Val(1)))
        @test size(y) == (1, length(x))
        @test vec(y) == x

        y = @inferred(OptimalTransport.add_singleton(x, Val(2)))
        @test size(y) == (length(x), 1)
        @test vec(y) == x

        x = rand(3, 4)
        y = @inferred(OptimalTransport.add_singleton(x, Val(1)))
        @test size(y) == (1, size(x, 1), size(x, 2))
        @test vec(y) == vec(x)

        y = @inferred(OptimalTransport.add_singleton(x, Val(2)))
        @test size(y) == (size(x, 1), 1, size(x, 2))
        @test vec(y) == vec(x)

        y = @inferred(OptimalTransport.add_singleton(x, Val(3)))
        @test size(y) == (size(x, 1), size(x, 2), 1)
        @test vec(y) == vec(x)
    end

    @testset "dot_matwise" begin
        l, m, n = 4, 5, 3
        x = rand(l, m)
        y = rand(l, m)
        @test OptimalTransport.dot_matwise(x, y) == dot(x, y)

        y = rand(l, m, n)
        @test OptimalTransport.dot_matwise(x, y) ==
              mapreduce(vcat, (view(y, :, :, i) for i in axes(y, 3))) do yi
            dot(x, yi)
        end
        @test OptimalTransport.dot_matwise(y, x) == OptimalTransport.dot_matwise(x, y)
    end

    @testset "checksize2" begin
        x = rand(5)
        y = rand(10)
        @test OptimalTransport.checksize2(x, y) === ()

        d = 4
        for (size2_x, size2_y) in
            (((), (d,)), ((1,), (d,)), ((d,), ()), ((d,), (1,)), ((d,), (d,)))
            x = rand(5, size2_x...)
            y = rand(10, size2_y...)
            @test OptimalTransport.checksize2(x, y) == (d,)
        end

        x = rand(5, 4)
        y = rand(10, 3)
        @test_throws DimensionMismatch OptimalTransport.checksize2(x, y)
    end

    @testset "checkbalanced" begin
        mass = rand()

        x1 = rand(20)
        x1 .*= mass / sum(x1)
        y1 = rand(30)
        y1 .*= mass / sum(y1)
        @test OptimalTransport.checkbalanced(x1, y1) === nothing
        @test OptimalTransport.checkbalanced(y1, x1) === nothing
        @test_throws ArgumentError OptimalTransport.checkbalanced(rand() .* x1, y1)
        @test_throws ArgumentError OptimalTransport.checkbalanced(x1, rand() .* y1)

        y2 = rand(30, 5)
        y2 .*= mass ./ sum(y2; dims=1)
        @test OptimalTransport.checkbalanced(x1, y2) === nothing
        @test OptimalTransport.checkbalanced(y2, x1) === nothing
        @test_throws ArgumentError OptimalTransport.checkbalanced(rand() .* x1, y2)
        @test_throws ArgumentError OptimalTransport.checkbalanced(
            x1, y2 .* hcat(rand(), ones(1, size(y2, 2) - 1))
        )

        x2 = rand(20, 5)
        x2 .*= mass ./ sum(x2; dims=1)
        @test OptimalTransport.checkbalanced(x2, y2) === nothing
        @test OptimalTransport.checkbalanced(y2, x2) === nothing
        @test_throws ArgumentError OptimalTransport.checkbalanced(
            x2 .* hcat(ones(1, size(x2, 2) - 1), rand()), y2
        )
        @test_throws ArgumentError OptimalTransport.checkbalanced(
            x2, y2 .* hcat(rand(), ones(1, size(y2, 2) - 1))
        )
    end
end
