include("../../src/Adjoints.jl")

module Michaelis

using ArgParse, Adjoints, Distributions, CatViews.CatView

export julia_main

include("../../util/optparser.jl")
include("../../util/experiment_ccompile.jl")
include("model.jl")

function twin_experiment_obs_m_logging!( # twin experiment with true and obs data logging
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
    a.obs[1,:,:] .= NaN
    for _replicate in 1:replicates
        a.obs[2,:,_replicate] .= view(a.x, 2, :) .+ rand(d, a.steps+1)
        for _i in 1:obs_iteration:a.steps
            for _k in 1:obs_iteration-1
                a.obs[2, _i + _k, _replicate] = NaN
            end
        end
        writedlm(dir * "observed$_replicate.tsv", view(a.obs, :, :, _replicate)')
    end
    twin_experiment!(dir, a, model, true_params, dists, trials)
end

Base.@ccallable function julia_main(args::Vector{String})::Cint
    parsed_args = parse_options(args)
    twin_experiment!(dxdt!, jacobian!, hessian!; parsed_args...)

    return 0
end

end
