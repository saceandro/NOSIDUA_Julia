@views function covariance_from_θ0!(a::Adjoint{N,L,T}, m::Model{N,L,T}, θ0) where {N,L,T<:AbstractFloat}
    initialize!(a, θ0)
    orbit!(a, m)
    gradient!(a, m)
    covariance!(a, m)
end

@views function assimilate!(a::Adjoint{N, L, T}, m::Model{N, L, T}, dists, trials=10) where {N,L,T}
    mincost = Inf
    local minres # we cannot define a new variable within for loop.
    for _i in 1:trials
        x0_p = rand.(dists)
        try
            res = minimize!(x0_p, a, m)
            println(STDERR, "trial $_i: $(res.minimum)")
            if res.minimum < mincost
                mincost = res.minimum
                minres = res
            end
        catch y
            println(STDERR, "trial $_i: minimization failed: $y")
        end
    end
    if !isfinite(mincost)
        error(STDERR, "All the $trials assimilation trials failed.")
    end
    return covariance_from_θ0!(a, m, minres.minimizer), minres
end
