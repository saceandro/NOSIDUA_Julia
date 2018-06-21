include("../../src/AdjointsConstraint.jl")

module Michaelis

using ArgParse, AdjointsConstraint, Distributions, CatViews.CatView, Juno

export julia_main

include("../../util/check_args.jl")
include("../../util/experiment_ccompile.jl")
include("model.jl")

@views function twin_experiment_obs_m!( # twin experiment with true and obs data logging
    dxdt!::Function,
    jacobian!::Function,
    jacobian0!::Function,
    hessian!::Function,
    hessian0!::Function,
    hessian00!::Function;
    dir = nothing,
    true_params = nothing,
    initial_lower_bounds = nothing,
    initial_upper_bounds = nothing,
    pseudo_obs = nothing,
    pseudo_obs_var = nothing,
    obs_variance = nothing,
    obs_iteration = nothing,
    dt = nothing,
    spinup = nothing,
    duration = nothing,
    generation_seed = nothing,
    trials = nothing,
    replicates = nothing,
    iter = nothing,
    x0 = nothing
    )

    L = length(initial_lower_bounds)
    N = L - length(true_params)
    model = Model(typeof(dt), N, L, dxdt!, jacobian!, jacobian0!, hessian!, hessian0!, hessian00!)
    # srand(generation_seed)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L]
    # x0 = rand.(view(dists, 1:N))
    a = Adjoint(dt, duration, pseudo_obs, pseudo_obs_var, x0, copy(true_params), replicates)
    orbit!(a, model)
    dir *= "/true_params_$(join(true_params, "_"))/initial_lower_bounds_$(join(initial_lower_bounds, "_"))/initial_upper_bounds_$(join(initial_upper_bounds, "_"))/pseudo_obs_$(join(pseudo_obs, "_"))/pseudo_obs_var_$(join(pseudo_obs_var, "_"))/spinup_$spinup/trials_$trials/obs_variance_$obs_variance/obs_iteration_$obs_iteration/dt_$dt/duration_$duration/replicates_$replicates/iter_$iter/"
    srand(hash([true_params, initial_lower_bounds, initial_upper_bounds, pseudo_obs, pseudo_obs_var, obs_variance, obs_iteration, dt, spinup, duration, generation_seed, trials, replicates, iter]))
    d = Normal.(0., sqrt.([obs_variance*10., obs_variance]))
    obs = Array{typeof(dt)}(N, a.steps+1, replicates)

    # for _j in 1:N
    #     for _replicate in 1:replicates
    #         obs[_j,:,_replicate] .= a.x[_j,:] .+ rand(d[_j], a.steps+1)
    #         for _i in 1:obs_iteration:a.steps
    #             for _k in 1:obs_iteration-1
    #                 obs[_j, _i + _k, _replicate] = NaN
    #             end
    #         end
    #     end
    # end

    obs[1,:,:] .= NaN
    for _replicate in 1:replicates
        obs[2,:,_replicate] .= a.x[2,:] .+ rand(d[2], a.steps+1)
        for _i in 1:obs_iteration:a.steps
            for _k in 1:obs_iteration-1
                obs[2, _i + _k, _replicate] = NaN
            end
        end
    end

    obs_mean_var!(a, model, obs)
    twin_experiment!(dir, a, model, true_params, initial_lower_bounds, initial_upper_bounds, dists, trials)
end

Base.@ccallable function julia_main(args::Vector{String})::Cint
    settings = ArgParseSettings("Adjoint method",
                                prog = first(splitext(basename(@__FILE__))),
                                version = "$(first(splitext(basename(@__FILE__)))) version 0.1",
                                add_version = true,
                                autofix_names = true)

    @add_arg_table settings begin
        "--dir", "-d"
            help = "output directory"
            default = "result"
        "--true-params", "-p"
            help = "true parameters"
            arg_type = Float64
            nargs = '*'
            default = [0.5, 2.]
        "--initial-lower-bounds", "-l"
            help = "lower bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = [0., 0., 0., 1.]
        "--initial-upper-bounds", "-u"
            help = "upper bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = [2., 2., 1., 3.]
        "--pseudo-obs"
            help = "#pseudo observations"
            arg_type = Int
            nargs = '+'
            default = [0, 0]
        "--pseudo-obs-var"
            help = "variance of pseudo observations"
            arg_type = Float64
            nargs = '+'
            default = [1., 1.]
        # "--obs-variance"
        #     help = "observation variance"
        #     arg_type = Float64
        #     nargs = '+'
        #     default = [0.1, 0.01]
        "--obs-variance"
            help = "observation variance"
            arg_type = Float64
            default = 0.001
        "--obs-iteration"
            help = "observation iteration"
            arg_type = Int
            default = 5
        "--dt"
            help = "Δt"
            arg_type = Float64
            default = 1.
        "--spinup"
            help = "spinup"
            arg_type = Float64
            default = 0.
        "--duration", "-t"
            help = "assimilation duration"
            arg_type = Float64
            default = 100.
        # "--generation-seed", "-s"
        #     help = "seed for orbit generation"
        #     arg_type = Int
        #     default = 0
        "--trials"
            help = "#trials for gradient descent initial value"
            arg_type = Int
            default = 5
        "--replicates"
            help = "#replicates"
            arg_type = Int
            default = 1
        "--iter"
            help = "#iterations"
            arg_type = Int
            default = 1
        "--x0"
            help = "initial x"
            arg_type = Float64
            nargs = '+'
            default = [10./11., 40./43.]
    end

    parsed_args = parse_args(args, settings; as_symbols=true) # ARGS is needed for static compilation; Otherwise, global ARGS is used.
    check_args(settings; parsed_args...)
    twin_experiment_obs_m!(dxdt!, jacobian!, jacobian0!, hessian!, hessian0!, hessian00!; parsed_args...)

    return 0
end

end
