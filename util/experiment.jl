include("../src/Adjoints.jl")
using Adjoints, Distributions #, DataFrames, PlotlyJS, Gadfly

@views function assimilate_experiment!(model::Model{N,L}, observed_file, obs_variance, dt, dists, trials=10) where {N,L}
    obs = readdlm(observed_file)'
    steps = size(obs,1)
    a = Adjoint(dt, obs_variance, obs, L-N)
    assimilate!(a, model, dists, trials)
end

@views function twin_experiment_given_assim!(assimilation_results, minimum, true_params, tob)
    println("mincost:\t", minimum)
    println("θ:\t", assimilation_results.θ)
    println("ans:\t", CatViews.CatView(tob[:,1], true_params))
    println("diff:\t", assimilation_results.θ .- CatViews.CatView(tob[:,1], true_params))
    if !(isnull(assimilation_results.stddev))
        println("CI:\t", get(assimilation_results.stddev))
    end
end

@views function twin_experiment_given_data!(model::Model{N,L,T}, observed_file::String, obs_variance::T, dt::T, true_params::AbstractVector{T}, tob::AbstractMatrix{T}, dists, trials=10) where {N,L,T<:AbstractFloat}
    assimres, minres = assimilate_experiment!(model, observed_file, obs_variance, dt, dists, trials)
    twin_experiment_given_assim!(assimres, minres.minimum, true_params, tob)
end

@views twin_experiment_given_file!(model::Model{N,L,T}, observed_file::String, obs_variance::T, dt::T, true_params::AbstractVector{T}, true_file::String, dists, trials=10) where {N,L,T<:AbstractFloat} = twin_experiment_given_data!(model, observed_file, obs_variance, dt, true_params, readdlm(true_file)', dists, trials)
