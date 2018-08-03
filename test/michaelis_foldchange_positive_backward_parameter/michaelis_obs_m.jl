include("../../src/AdjointsGivenZeroStateBackward.jl")

module Michaelis

using ArgParse, AdjointsGivenZeroStateBackward, Distributions, CatViews.CatView

export julia_main

include("../../util/check_args_given_zero_state.jl")
include("../../util/experiment_ccompile_given_zero_state.jl")
include("model.jl")

@views function twin_experiment_obs_m!( # twin experiment with true and obs data logging
    dxdt!::Function,
    jacobianx!::Function,
    jacobianp!::Function,
    hessianxx!::Function,
    hessianxp!::Function,
    hessianpp!::Function,
    observation::Function,
    d_observation::Function,
    dd_observation::Function,
    inv_observation::Function;
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
    trials = nothing,
    newton_maxiter = nothing,
    newton_tol = nothing,
    regularization_coefficient = nothing,
    replicates = nothing,
    iter = nothing,
    x0 = nothing
    )

    N = length(x0)
    L = N + length(true_params)
    model = Model(typeof(dt), N, L, dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, observation, d_observation, dd_observation, inv_observation)
    # srand(generation_seed)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L-N]
    # x0 = rand.(view(dists, 1:N))
    a = Adjoint(dt, duration, pseudo_obs, pseudo_obs_var, x0, copy(true_params), replicates, newton_maxiter, newton_tol, regularization_coefficient)
    orbit!(a, model)
    dir *= "/true_params_$(join_digits10(true_params))/initial_lower_bounds_$(join_digits10(initial_lower_bounds))/initial_upper_bounds_$(join_digits10(initial_upper_bounds))/pseudo_obs_$(join_digits10(pseudo_obs))/pseudo_obs_var_$(join_digits10(pseudo_obs_var))/spinup_$(digits10(spinup))/trials_$(digits10(trials))/newton_maxiter_$(digits10(newton_maxiter))/newton_tol_$(digits10(newton_tol))/regularization_coefficient_$(digits10(regularization_coefficient))/obs_variance_$(digits10(obs_variance))/obs_iteration_$(digits10(obs_iteration))/dt_$(digits10(dt))/duration_$(digits10(duration))/replicates_$(digits10(replicates))/iter_$(digits10(iter))/"
    srand(hash([true_params, initial_lower_bounds, initial_upper_bounds, pseudo_obs, pseudo_obs_var, obs_variance, obs_iteration, dt, spinup, duration, trials, newton_maxiter, newton_tol, replicates, iter]))
    d = Normal.(0., sqrt.([obs_variance*10., obs_variance]))
    obs = Array{typeof(dt)}(N, a.steps+1, replicates)

    # for _j in 1:N
    #     for _replicate in 1:replicates
    #         obs[_j,:,_replicate] .= model.observation.(a.x[_j,:]) .+ rand(d[_j], a.steps+1)
    #         for _i in 1:obs_iteration:a.steps
    #             for _k in 1:obs_iteration-1
    #                 obs[_j, _i + _k, _replicate] = NaN
    #             end
    #         end
    #     end
    # end

    obs[1,:,:] .= NaN
    for _replicate in 1:replicates
        obs[2,:,_replicate] .= model.observation.(a.x[2,:]) .+ rand(d[2], a.steps+1)
        for _i in 1:obs_iteration:a.steps
            for _k in 1:obs_iteration-1
                obs[2, _i + _k, _replicate] = NaN
            end
        end
    end
    obs[:,1,:] .= NaN
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
            # default = [0.5, 1., 5., 4.]
            default = [log(0.3), log(1.), log(5.), log(0.15)]
        "--initial-lower-bounds", "-l"
            help = "lower bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = [log(0.1), log(0.1), log(1.), log(0.1)]
        "--initial-upper-bounds", "-u"
            help = "upper bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = [log(1.), log(10.), log(10.), log(1.)]
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
            default = 0.1
        "--spinup"
            help = "spinup"
            arg_type = Float64
            default = 0.
        "--duration", "-t"
            help = "assimilation duration"
            arg_type = Float64
            default = 240.
        # "--generation-seed", "-s"
        #     help = "seed for orbit generation"
        #     arg_type = Int
        #     default = 0
        "--trials"
            help = "#trials for gradient descent initial value"
            arg_type = Int
            default = 20
        "--newton-maxiter"
            help = "#maxiter for newton's method"
            arg_type = Int
            default = 100
        "--newton-tol"
            help = "newton method toralence"
            arg_type = Float64
            default = 1e-8
        "--regularization-coefficient"
            help = "regularization coefficient"
            arg_type = Float64
            default = 1.
        "--replicates"
            help = "#replicates"
            arg_type = Int
            default = 100
        "--iter"
            help = "#iterations"
            arg_type = Int
            default = 1
        "--x0"
            help = "initial x"
            arg_type = Float64
            nargs = '+'
            default = [0., 0.]
    end

    parsed_args = parse_args(args, settings; as_symbols=true) # ARGS is needed for static compilation; Otherwise, global ARGS is used.
    check_args(settings; parsed_args...)
    twin_experiment_obs_m!(dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, observation, d_observation, dd_observation, inv_observation; parsed_args...)

    return 0
end

end
