function covariance_from_θ0!(a::Adjoint{N,L,K,T}, m::Model{N,L,T}, θ0) where {N,L,K,T<:AbstractFloat}
    initialize!(a, θ0)
    orbit!(a, m)
    gradient!(a, m, Vector{T}(L))
    covariance!(a, m)
end

function printres(io, res)
    println(io, res)
end

function assimilate!(a::Adjoint{N,L,K,T}, m::Model, initial_lower_bounds, initial_upper_bounds, dists, trials=10) where {N,L,K,T<:AbstractFloat}
    mincost = Inf
    local minres # we cannot define a new variable within for loop.
    for _i in 1:trials
        θ0 = rand.(dists)
        try
            res = minimize!(θ0, a, m)

            println(STDERR, "trial $_i: $(res.minimizer)\t$(res.minimum)")
            println(STDERR, res)
            # if (res.minimum < mincost) && all(initial_lower_bounds .<= res.minimizer) && all(res.minimizer .<= initial_upper_bounds)
            # if (res.minimum < mincost) && (res.x_converged || res.f_converged || res.g_converged)  && all(res.minimizer .> 0.)
            # if (res.minimum < mincost) && all(res.minimizer .> 0.)
            # if res.minimum < mincost
            if (res.minimum < mincost) && (res.x_converged || res.f_converged || res.g_converged)
                mincost = res.minimum
                minres = res
            end
        catch y
            println(STDERR, "trial $_i: minimization failed: $y")
        end
    end
    if !isfinite(mincost)
        return AssimilationResults(L, T), Nullable{Optim.MultivariateOptimizationResults}()
        # error(STDERR, "All the $trials assimilation trials failed.")
    end
    return covariance_from_θ0!(a, m, minres.minimizer), minres
end
