nanzero(x) = isnan(x) ? zero(x) : x

digits10(x) = map(x -> @sprintf("%.10f", x), x)
join_digits10(x) = join(digits10(x), "_")

digits3(x) = map(x -> @sprintf("%.3f", x), x)
join_digits3(x) = join(digits3(x), "_")

@views function obs_mean_var!(a::Adjoint{N,L,R,U}, m::Model{N,L,R,U}, obs) where {N,L,R,U}
    all!(a.finite, isfinite.(obs))
    a.obs_mean = reshape(mean(obs; dims=3), U, a.steps+1)
    # a.obs_filterd_var = reshape(sum(reshape(reshape(var(obs, 3; corrected=false), N, a.steps+1)[a.finite], N, :), 2), N)
    for _j in 1:U
        a.obs_filterd_var[_j] = mapreduce(nanzero, +, var(obs[_j,:,:]; dims=2, corrected=false))
        a.Nobs[_j] += count(isfinite.(obs[_j,:,:]))
    end
    nothing
end

@views function write_twin_experiment_result(dir, assimilation_results::AssimilationResults{L}, minimum, true_params, tob) where {L}
    mkpath(dir)
    if assimilation_results.θ == nothing
        writedlm(dir * "estimates.tsv", reshape(CatView(fill(NaN, L), fill(NaN, L)), L, 2))
        return nothing
    end
    # L = length(assimilation_results.θ)
    println(stderr, "mincost:\t", minimum)
    println(stderr, "θ:\t", assimilation_results.θ)
    println(stderr, "ans:\t", CatView(tob[:,1], true_params))
    diff = assimilation_results.θ .- CatView(tob[:,1], true_params)
    println(stderr, "diff:\t", diff)
    println(sqrt(mapreduce(abs2, +, diff) / L)) # output RSME to stdout

    if assimilation_results.precision != nothing
        println(stderr, "precision:\t", assimilation_results.precision)
    end

    if assimilation_results.stddev == nothing
        writedlm(dir * "estimates.tsv", reshape(CatView(diff, fill(NaN, L)), L, 2))
    else
        println(stderr, "CI:\t", assimilation_results.stddev)
        writedlm(dir * "estimates.tsv", reshape(CatView(diff, assimilation_results.stddev), L, 2))
    end
    println(stderr, "obs variance:\t", assimilation_results.obs_variance)
    nothing
end

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

# function twin_experiment!(outdir::String, model::Model{N,L,T}, obs::AbstractArray{T,3}, obs_variance::T, dt::T, true_params::AbstractVector{T}, tob::AbstractMatrix{T}, dists, trials=10) where {N,L,T<:AbstractFloat}
#     a = Adjoint(dt, obs_variance, obs, L-N)
#     assimres, minres = assimilate!(a, model, dists, trials)
#     write_twin_experiment_result(outdir, assimres, minres.minimum, true_params, tob)
# end

# twin_experiment!(outdir::String, model::Model{N,L,T}, obs_variance::T, obs_iteration::Int, dt::T, true_params::AbstractVector{T}, true_file::String, dists, replicates::Int, iter::Int, trials=10) where {N,L,T} = twin_experiment!(outdir, model, obs_variance, obs_iteration, dt, true_params, readdlm(true_file)', dists, replicates, iter, trials)

@views function twin_experiment!(
    dxdt!::Function,
    jacobian!::Function,
    hessian!::Function;
    dir = nothing,
    true_params = nothing,
    initial_lower_bounds = nothing,
    initial_upper_bounds = nothing,
    obs_variance = nothing,
    obs_iteration = nothing,
    dt = nothing,
    spinup = nothing,
    duration = nothing,
    generation_seed = nothing,
    trials = nothing,
    replicates = nothing,
    iter = nothing
    )

    L = length(initial_lower_bounds)
    N = L - length(true_params)
    model = Model(typeof(dt), N, L, dxdt!, jacobian!, hessian!)
    srand(generation_seed)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L]
    x0 = rand.(view(dists, 1:N))
    a = Adjoint(dt, duration, copy(obs_variance), x0, copy(true_params), replicates)
    orbit!(a, model)
    srand(hash([true_params, initial_lower_bounds, initial_upper_bounds, obs_variance, obs_iteration, dt, spinup, duration, generation_seed, trials, replicates, iter]))
    d = Normal.(0., sqrt.(obs_variance))
    for _replicate in 1:replicates
        for _i in 1:N
            a.obs[_i,:,_replicate] .= a.x[_i,:] .+ rand(d[_i], a.steps+1)
        end
        for _i in 1:obs_iteration:a.steps
            for _k in 1:obs_iteration-1
                a.obs[:, _i + _k, _replicate] .= NaN
            end
        end
    end
    dir *= "/true_params_$(join(true_params, "_"))/initial_lower_bounds_$(join(initial_lower_bounds, "_"))/initial_upper_bounds_$(join(initial_upper_bounds, "_"))/spinup_$spinup/generation_seed_$generation_seed/trials_$trials/obs_variance_$(join(obs_variance, "_"))/obs_iteration_$obs_iteration/dt_$dt/duration_$duration/replicates_$replicates/iter_$iter/"
    twin_experiment!(dir, a, model, true_params, initial_lower_bounds, initial_upper_bounds, dists, trials)
end

function twin_experiment_logging!( # twin experiment with true and obs data logging
    dxdt!::Function,
    jacobian!::Function,
    hessian!::Function;
    dir = nothing,
    true_params = nothing,
    initial_lower_bounds = nothing,
    initial_upper_bounds = nothing,
    obs_variance = nothing,
    obs_iteration = nothing,
    dt = nothing,
    spinup = nothing,
    duration = nothing,
    generation_seed = nothing,
    trials = nothing,
    replicates = nothing,
    iter = nothing
    )

    L = length(initial_lower_bounds)
    N = L - length(true_params)
    model = Model(typeof(obs_variance), N, L, dxdt!, jacobian!, hessian!)
    srand(generation_seed)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L]
    x0 = rand.(view(dists, 1:N))
    a = Adjoint(dt, duration, obs_variance, x0, copy(true_params), replicates)
    orbit!(a, model)
    dir *= "/true_params_$(join(true_params, "_"))/initial_lower_bounds_$(join(initial_lower_bounds, "_"))/initial_upper_bounds_$(join(initial_upper_bounds, "_"))/spinup_$spinup/generation_seed_$generation_seed/trials_$trials/obs_variance_$obs_variance/obs_iteration_$obs_iteration/dt_$dt/duration_$duration/replicates_$replicates/iter_$iter/"
    mkpath(dir)
    writedlm(dir * "true.tsv", a.x')
    srand(hash([true_params, initial_lower_bounds, initial_upper_bounds, obs_variance, obs_iteration, dt, spinup, duration, generation_seed, trials, replicates, iter]))
    d = Normal(0., sqrt(obs_variance))
    for _replicate in 1:replicates
        a.obs[:,:,_replicate] .= a.x .+ rand(d, N, a.steps+1)
        for _i in 1:obs_iteration:a.steps
            for _k in 1:obs_iteration-1
                a.obs[:, _i + _k, _replicate] .= NaN
            end
        end
        writedlm(dir * "observed$_replicate.tsv", view(a.obs, :, :, _replicate)')
    end
    twin_experiment!(dir, a, model, true_params, dists, trials)
end


function twin_experiment!(outdir::String, a::Adjoint{N,L,R,U,K,T}, model::Model{N,L,R,U,T}, true_params::AbstractVector{T}, initial_lower_bounds::AbstractVector{T}, initial_upper_bounds::AbstractVector{T}, dists, trials=10) where {N,L,R,U,K,T<:AbstractFloat}
    tob = deepcopy(a.x) # fixed bug. copy() of >=2 dimensional array is implemented as reference. Thus, deepcopy() is needed.
    println(stderr, "====================================================================================================================")
    println(stderr, outdir)
    assimres, minres = assimilate!(a, model, initial_lower_bounds, initial_upper_bounds, dists, trials)
    if assimres.θ == nothing
        write_twin_experiment_result(outdir, assimres, zero(T), true_params, tob)
    else
        write_twin_experiment_result(outdir, assimres, minres.minimum, true_params, tob)
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
