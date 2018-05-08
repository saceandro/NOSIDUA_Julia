@views function write_twin_experiment_result(pref, assimilation_results, minimum, true_params, tob)
    mkpath(pref)
    N = length(assimilation_results.θ)
    println(STDERR, "mincost:\t", minimum)
    println(STDERR, "θ:\t", assimilation_results.θ)
    println(STDERR, "ans:\t", CatView(tob[:,1], true_params))
    diff = assimilation_results.θ .- CatView(tob[:,1], true_params)
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

# function twin_experiment!(model::Model{N,L,T}, obs::AbstractArray{T,3}, obs_variance::T, dt::T, true_params::AbstractVector{T}, tob::AbstractMatrix{T}, dists, trials=10) where {N,L,T<:AbstractFloat}
#     replicates = size(obs, 3)
#     pref = "results/replicate_$replicates/"
#     a = Adjoint(dt, obs_variance, obs, L-N)
#     assimres, minres = assimilate!(a, model, dists, trials)
#     write_twin_experiment_result(pref, assimres, minres.minimum, true_params, tob)
# end
#
# function twin_experiment!(model::Model{N,L,T}, observed_files::Tuple, obs_variance::T, dt::T, true_params::AbstractVector{T}, true_file::String, dists, trials=10) where {N,L,T<:AbstractFloat}
#     obs1 = readdlm(observed_files[1])'
#     steps = size(obs1,2)
#     K = length(observed_files)
#     obs = Array{T}(N, steps, K)
#     obs[:,:,1] .= obs1
#     for _replicate in 2:K
#         obs[:,:,_replicate] .= readdlm(observed_files[_replicate])'
#     end
#     twin_experiment!(model, obs, obs_variance, dt, true_params, readdlm(true_file)', dists, trials)
# end
#
# function twin_experiment!(outdir::String, model::Model{N,L,T}, obs::AbstractArray{T,3}, obs_variance::T, dt::T, true_params::AbstractVector{T}, tob::AbstractMatrix{T}, dists, trials=10) where {N,L,T<:AbstractFloat}
#     a = Adjoint(dt, obs_variance, obs, L-N)
#     assimres, minres = assimilate!(a, model, dists, trials)
#     write_twin_experiment_result(outdir, assimres, minres.minimum, true_params, tob)
# end
#
# function twin_experiment!(outdir::String, model::Model{N,L,T}, obs_variance::T, obs_iteration::Int, dt::T, true_params::AbstractVector{T}, tob::AbstractMatrix{T}, dists, replicates::Int, iter::Int, trials=10) where {N,L,T}
#     steps = size(tob, 2)
#     srand(hash([replicates, iter]))
#     d = Normal(0., sqrt(obs_variance))
#     obs = similar(tob, N, steps, replicates)
#     for _replicate in 1:replicates
#         obs[:,:,_replicate] .= tob .+ rand(d, N, steps)
#         for _i in 1:obs_iteration:steps-1
#             for _k in 1:obs_iteration-1
#                 obs[:, _i + _k, _replicate] .= NaN
#             end
#         end
#     end
#     twin_experiment!(outdir * "replicate_$replicates/iter_$iter/", model, obs, obs_variance, dt, true_params, tob, dists, trials)
# end
#
# twin_experiment!(outdir::String, model::Model{N,L,T}, obs_variance::T, obs_iteration::Int, dt::T, true_params::AbstractVector{T}, true_file::String, dists, replicates::Int, iter::Int, trials=10) where {N,L,T} = twin_experiment!(outdir, model, obs_variance, obs_iteration, dt, true_params, readdlm(true_file)', dists, replicates, iter, trials)

# function twin_experiment!(model::Model{N,L}, parsed_args) where {N,L}
#     args2varname(parsed_args)
#     srand(generation_seed)
#     dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L]
#     x0 = rand.(view(dists, 1:N))
#     a = Adjoint(dt, duration, obs_variance, x0, copy(true_params), replicates)
#     orbit!(a, model)
#     srand(hash(parsed_args))
#     d = Normal(0., sqrt(obs_variance))
#     for _replicate in 1:replicates
#         a.obs[:,:,_replicate] .= a.x .+ rand(d, N, a.steps+1)
#         for _i in 1:obs_iteration:a.steps
#             for _k in 1:obs_iteration-1
#                 a.obs[:, _i + _k, _replicate] .= NaN
#             end
#         end
#     end
#     twin_experiment!(dir, a, model, true_params, dists, trials)
# end

# twin_experiment!(
#     model::Model{N,L};
#     dir = "result1/",
#     dimension = 5,
#     true_params = [8., 1.],
#     initial_lower_bounds = [-10.,-10.,-10.,-10.,-10.,0.,0.],
#     initial_upper_bounds = [10.,10.,10.,10.,10.,16.,2.],
#     obs_variance = 1.,
#     obs_iteration = 5,
#     dt = 0.01,
#     spinup = 73.,
#     duration = 1.,
#     generation_seed = 0,
#     trials = 50,
#     replicates = 1,
#     iter = 1
#     ) where {N,L} = twin_experiment!(
#                         hash([dimension, true_params, initial_lower_bounds, initial_upper_bounds, obs_variance, obs_iteration, dt, spinup, duration, generation_seed, trials, replicates, iter]),
#                         dir,
#                         model,
#                         obs_variance,
#                         obs_iteration,
#                         dt,
#                         spinup,
#                         duration,
#                         generation_seed,
#                         true_params,
#                         initial_lower_bounds,
#                         initial_upper_bounds,
#                         replicates,
#                         iter,
#                         trials
#                     )

function twin_experiment!(
    model::Model{N,L};
    dir = "result1/",
    dimension = 5,
    true_params = [8., 1.],
    initial_lower_bounds = [-10.,-10.,-10.,-10.,-10.,0.,0.],
    initial_upper_bounds = [10.,10.,10.,10.,10.,16.,2.],
    obs_variance = 1.,
    obs_iteration = 5,
    dt = 0.01,
    spinup = 73.,
    duration = 1.,
    generation_seed = 0,
    trials = 50,
    replicates = 1,
    iter = 1
    ) where {N,L}

    srand(generation_seed)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L]
    x0 = rand.(view(dists, 1:N))
    a = Adjoint(dt, duration, obs_variance, x0, copy(true_params), replicates)
    orbit!(a, model)
    srand(hash([dimension, true_params, initial_lower_bounds, initial_upper_bounds, obs_variance, obs_iteration, dt, spinup, duration, generation_seed, trials, replicates, iter]))
    d = Normal(0., sqrt(obs_variance))
    for _replicate in 1:replicates
        a.obs[:,:,_replicate] .= a.x .+ rand(d, N, a.steps+1)
        for _i in 1:obs_iteration:a.steps
            for _k in 1:obs_iteration-1
                a.obs[:, _i + _k, _replicate] .= NaN
            end
        end
    end
    twin_experiment!(dir, a, model, true_params, dists, trials)
end

# function twin_experiment!(model::Model{N,L}, parsed_args) where {N,L}
#     args2varname(parsed_args)
#     twin_experiment!(hash(parsed_args), dir, model, obs_variance, obs_iteration, dt, spinup, duration, generation_seed, true_params, initial_lower_bounds, initial_upper_bounds, replicates, iter, trials)
# end

# function twin_experiment!(model::Model{N,L}, s) where {N,L}
#     srand(s["generation_seed"])
#     dists = [Uniform(s["initial_lower_bounds"][i], s["initial_upper_bounds"][i]) for i in 1:L]
#     x0 = rand.(view(dists, 1:N))
#     a = Adjoint(s["dt"], s["duration"], s["obs_variance"], x0, copy(s["true_params"]), s["replicates"])
#     orbit!(a, model)
#     srand(hash(s))
#     d = Normal(0., sqrt(s["obs_variance"]))
#     for _replicate in 1:s["replicates"]
#         a.obs[:,:,_replicate] .= a.x .+ rand(d, N, a.steps+1)
#         for _i in 1:s["obs_iteration"]:a.steps
#             for _k in 1:s["obs_iteration"]-1
#                 a.obs[:, _i + _k, _replicate] .= NaN
#             end
#         end
#     end
#     twin_experiment!(s["dir"], a, model, s["true_params"], dists, s["trials"])
# end

# function twin_experiment!(args_hash, outdir::String, model::Model{N,L,T}, obs_variance::T, obs_iteration::Int, dt::T, spinup::T, total_T::T, generation_seed::Int, true_params::AbstractVector{T}, initial_lower_bounds::AbstractVector{T}, initial_upper_bounds::AbstractVector{T}, replicates::Int, iter::Int, trials=10) where {N,L,T}
#     srand(generation_seed)
#     dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L]
#     x0 = rand.(view(dists, 1:N))
#     a = Adjoint(dt, total_T, obs_variance, x0, copy(true_params), replicates)
#     orbit!(a, model)
#     srand(args_hash)
#     d = Normal(0., sqrt(obs_variance))
#     for _replicate in 1:replicates
#         a.obs[:,:,_replicate] .= a.x .+ rand(d, N, a.steps+1)
#         for _i in 1:obs_iteration:a.steps
#             for _k in 1:obs_iteration-1
#                 a.obs[:, _i + _k, _replicate] .= NaN
#             end
#         end
#     end
#     twin_experiment!(outdir, a, model, true_params, dists, trials)
# end

function twin_experiment!(outdir::String, a::Adjoint{N,L,K,T}, model::Model{N,L,T}, true_params::AbstractVector{T}, dists, trials=10) where {N,L,K,T<:AbstractFloat}
    tob = copy(a.x)
    assimres, minres = assimilate!(a, model, dists, trials)
    write_twin_experiment_result(outdir, assimres, minres.minimum, true_params, tob)
end

function twin_experiment_iter!(outdir::String, model::Model{N,L}, true_params, obs_variance, obs_iteration, dt, spinup, T, seed, replicates, dists, trials=10, iter=10) where {N,L}
    srand(seed)
    x0 = rand.(dists[1:N])
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

@views function generate_true_data!(pref::String, model::Model{N}, true_params, dt, spinup, T, generation_seed, x0_dists) where {N}
    srand(generation_seed)
    x0 = rand.(x0_dists)
    a = Adjoint(dt, T, 1., x0, true_params) # 1. means obs_variance. defined as 1 though not used.
    orbit!(a, model)
    mkpath(pref)
    writedlm(pref * "seed_$generation_seed.tsv", a.x')
end
