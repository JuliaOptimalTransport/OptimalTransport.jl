struct SinkhornDivergence{A<:Sinkhorn,B<:Sinkhorn,C<:Sinkhorn}
    algμν::A
    algμμ::B
    algνν::C
end

"""
    sinkhorn_divergence(
        μ::AbstractVecOrMat,
        ν::AbstractVecOrMat,
        C,
        ε,
        alg::SinkhornDivergence=SinkhornDivergence(
            SinkhornGibbs(), SymmetricSinkhornGibbs(), SymmetricSinkhornGibbs()
        );
        kwargs...,
    )

Compute the Sinkhorn Divergence between finite discrete
measures `μ` and `ν` with respect to a common cost matrix `C`,
entropic regularization parameter `ε` and algorithm `alg`. 

In the default case where `regularization = false`, the Sinkhorn Divergence is that of [^GPC18] and is computed as
```math
\\operatorname{S}_{ε}(μ,ν) := \\operatorname{W}_{ε}(μ,ν)
- \\frac{1}{2}(\\operatorname{W}_{ε}(μ,μ) + \\operatorname{W}_{ε}(ν,ν)),
```
and ``\\operatorname{W}_{ε}`` is defined as
```math
\\operatorname{W}_{ε}(μ, ν) = \\langle C, γ^\\star \\rangle,
```
where ``γ^\\star`` is the entropy-regularised transport plan between `μ` and `ν`. 
For `regularization = true`, the Sinkhorn Divergence is that of [^FeydyP19] and is computed as above 
where ``\\operatorname{W}_{ε}`` is replaced by ``\\operatorname{OT}_{ε}``, the entropy-regularised optimal transport 
cost with regulariser penalty. 

The default algorithm for computing the term ``\\operatorname{W}_{ε}(μ, ν)`` is the `SinkhornGibbs` algorithm.
For the terms ``\\operatorname{W}_{ε}(μ, μ)`` and ``\\operatorname{W}_{ε}(ν, ν)``, the symmetric fixed point iteration of [^FeydyP19] is used. 
Alternatively, a pre-computed optimal transport `plan` between `μ` and `ν` may be provided.  

[^GPC18]: Aude Genevay, Gabriel Peyré, Marco Cuturi, Learning Generative Models with Sinkhorn Divergences, Proceedings of the Twenty-First International Conference on Artficial Intelligence and Statistics, (AISTATS) 21, 2018
[^FeydyP19]: Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé, and Gabriel Peyré. Interpolating between optimal transport and mmd using sinkhorn divergences. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 2681–2690. PMLR, 2019.
See also: [`sinkhorn2`](@ref)
"""
function sinkhorn_divergence(
    μ::AbstractVecOrMat,
    ν::AbstractVecOrMat,
    C,
    ε,
    alg::SinkhornDivergence=SinkhornDivergence(
        SinkhornGibbs(), SymmetricSinkhornGibbs(), SymmetricSinkhornGibbs()
    );
    kwargs...,
)
    return sinkhorn_divergence(μ, ν, C, C, C, ε, alg; kwargs...)
end

"""
    sinkhorn_divergence(
        μ,
        ν,
        Cμν,
        Cμ,
        Cν,
        ε,
        alg::SinkhornDivergence=SinkhornDivergence(
            SinkhornGibbs(), SymmetricSinkhornGibbs(), SymmetricSinkhornGibbs()
        );
        kwargs...,
    )

Compute the Sinkhorn Divergence between finite discrete
measures `μ` and `ν` with respect to the precomputed cost matrices `Cμν`,
`Cμμ` and `Cνν`, entropic regularization parameter `ε` and algorithm `alg`.

A pre-computed optimal transport `plan` between `μ` and `ν` may be provided.

See also: [`sinkhorn2`](@ref), [`sinkhorn_divergence`](@ref)
"""
function sinkhorn_divergence(
    μ,
    ν,
    Cμν,
    Cμ,
    Cν,
    ε,
    alg::SinkhornDivergence=SinkhornDivergence(
        SinkhornGibbs(), SymmetricSinkhornGibbs(), SymmetricSinkhornGibbs()
    );
    kwargs...,
)
    OTμν = sinkhorn2(μ, ν, Cμν, ε, alg.algμν; kwargs...)
    OTμμ = sinkhorn2(μ, Cμ, ε, alg.algμμ; kwargs...)
    OTνν = sinkhorn2(ν, Cν, ε, alg.algνν; kwargs...)
    return max.(0, OTμν .- (OTμμ .+ OTνν) / 2)
end
