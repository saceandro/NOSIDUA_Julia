@views function write_twin_experiment_result(pref, assimilation_results, minimum, true_params, tob)
    mkpath(pref)
    N = length(assimilation_results.θ)
    println(STDERR, "mincost:\t", minimum)
    println(STDERR, "θ:\t", assimilation_results.θ)
    println(STDERR, "ans:\t", [tob[:,1]; true_params])
    diff = assimilation_results.θ .- [tob[:,1]; true_params]
    println(STDERR, "diff:\t", diff)
    println(sqrt(mapreduce(abs2, +, diff) / N)) # output RMSE to STDOUT
    if !(isnull(assimilation_results.stddev))
        println(STDERR, "CI:\t", get(assimilation_results.stddev))
    end
    for _i in 1:N
        open(pref * "$_i.tsv", "w") do f
            println(f, "$(diff[_i])\t$(get(assimilation_results.stddev)[_i])")
        end
    end
    nothing
end

function twin_experiment!(model::Model{N,L,T}, obs::AbstractArray{T,3}, obs_variance::T, dt::T, true_params::AbstractVector{T}, tob::AbstractMatrix{T}, dists, trials=10) where {N,L,T<:AbstractFloat}
    replicates = size(obs, 3)
    pref = "results/replicate_$replicates/"
    a = Adjoint(dt, obs_variance, obs, L-N)
    assimres, minres = assimilate!(a, model, dists, trials)
    write_twin_experiment_result(pref, assimres, minres.minimum, true_params, tob)
end

function twin_experiment!(model::Model{N,L,T}, observed_files::Tuple, obs_variance::T, dt::T, true_params::AbstractVector{T}, true_file::String, dists, trials=10) where {N,L,T<:AbstractFloat}
    obs1 = readdlm(observed_files[1])'
    steps = size(obs1,2)
    K = length(observed_files)
    obs = Array{T}(N, steps, K)
    obs[:,:,1] .= obs1
    for _replicate in 2:K
        obs[:,:,_replicate] .= readdlm(observed_files[_replicate])'
    end
    twin_experiment!(model, obs, obs_variance, dt, true_params, readdlm(true_file)', dists, trials)
end

function twin_experiment!(outdir::String, model::Model{N,L,T}, obs::AbstractArray{T,3}, obs_variance::T, dt::T, true_params::AbstractVector{T}, tob::AbstractMatrix{T}, dists, trials=10) where {N,L,T<:AbstractFloat}
    a = Adjoint(dt, obs_variance, obs, L-N)
    assimres, minres = assimilate!(a, model, dists, trials)
    write_twin_experiment_result(outdir, assimres, minres.minimum, true_params, tob)
end

function twin_experiment_iter!(outdir::String, model::Model{N,L}, true_params, obs_variance, obs_iteration, dt, spinup, T, seed, replicates, dists, trials=10, iter=10) where {N,L}
    srand(seed)
    x0 = rand.(dists[1:5])
    a = Adjoint(dt, T, obs_variance, x0, true_params)
    orbit!(a, model)
    d = Normal(0., sqrt(obs_variance))
    for _it in 1:iter
        obs = Array{Float64}(N, a.steps+1, replicates)
        for _replicate in 1:replicates
            obs[:,:,_replicate] .= a.x .+ rand(d, N, a.steps+1)
            for _i in 1:obs_iteration:a.steps
                for _k in 1:obs_iteration-1
                    obs[:, _i + _k, _replicate] .= NaN
                end
            end
        end
        twin_experiment!(outdir * "replicate_$replicates/iter_$_it/", model, obs, obs_variance, dt, true_params, a.x, dists, trials)
    end
end
