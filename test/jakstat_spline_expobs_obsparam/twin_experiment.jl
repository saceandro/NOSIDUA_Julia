include("../../src/AdjointBackwardObsparam.jl")

module Jakstat

using AdjointsBackwardObsparam, CatViews.CatView, Optim, Distributions, ArgParse

export julia_main

include("../../util/check_args.jl")
include("../../util/experiment_ccompile_backward_obsparam.jl")
include("model.jl")

@views function twin_experiment_obs_m!( # twin experiment with true and obs data logging
    dxdt!::Function,
    jacobianx!::Function,
    jacobianp!::Function,
    hessianxx!::Function,
    hessianxp!::Function,
    hessianpp!::Function,
    calc_qdot!::Function,
    observation!::Function,
    observation_jacobianx!::Function,
    observation_jacobianr!::Function,
    observation_jacobianp!::Function,
    observation_hessianxx!::Function,
    observation_hessianxr!::Function,
    observation_hessianrr!::Function,
    observation_hessianpp!::Function;
    dir = nothing,
    number_of_params = nothing,
    number_of_obs_params = nothing,
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
    newton_maxiter = nothing,
    newton_tol = nothing,
    regularization_coefficient = nothing,
    replicates = nothing,
    iter = nothing,
    time_point = nothing,
    parameters = nothing
    )

    obs_variance_bak = copy(obs_variance)


    N = length(initial_lower_bounds) - number_of_params - number_of_obs_params
    L = N + number_of_params
    R = number_of_obs_params
    U = length(obs_variance)
    model = Model(typeof(dt), N, L, R, U, time_point, dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, calc_qdot!, observation!, observation_jacobianx!, observation_jacobianr!, observation_jacobianp!, observation_hessianxx!, observation_hessianxr!, observation_hessianrr!, observation_hessianpp!)
    srand(generation_seed)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L+R]
    x0 = rand.(dists[1:N])
    a = Adjoint(dt, duration, number_of_obs_params, pseudo_obs, pseudo_obs_var, x0, copy(true_params), replicates, newton_maxiter, newton_tol, regularization_coefficient)
    orbit_first!(a, model)
    tob = deepcopy(a.x)
    dir *= "/number_of_params_$(digits3(number_of_params))/number_of_obs_params_$(digits3(number_of_obs_params))/true_params_$(join_digits3(true_params))/initial_lower_bounds_$(join_digits3(initial_lower_bounds))/initial_upper_bounds_$(join_digits3(initial_upper_bounds))/pseudo_obs_$(join_digits3(pseudo_obs))/pseudo_obs_var_$(join_digits3(pseudo_obs_var))/obs_variance_$(join_digits3(obs_variance))/spinup_$(digits3(spinup))/trials_$(digits3(trials))/newton_maxiter_$(digits3(newton_maxiter))/newton_tol_$(digits3(newton_tol))/regularization_coefficient_$(digits3(regularization_coefficient))/time_point_$(join_digits3(time_point))/obs_iteration_$(digits3(obs_iteration))/dt_$(digits3(dt))/duration_$(digits3(duration))/replicates_$(digits3(replicates))/iter_$(digits3(iter))/"
    srand(hash([true_params, obs_variance_bak, obs_iteration, spinup, duration, generation_seed, replicates, iter, time_point]))

    d = Normal.(0., sqrt.(obs_variance_bak))
    obs = Array{typeof(dt)}(U, a.steps+1, replicates)

    obs[:, :, :] .= NaN
    for _replicate in 1:replicates
        for _l in 1:length(time_point)
            observation!(model, time_point[_l], a.x[:,Int(time_point[_l]/dt) + 1], a.p)
            obs[:, Int(time_point[_l]/dt) + 1, _replicate] .= model.observation .+ rand.(d)
            # obs[1,   Int(time_point[_l]/dt) + 1, _replicate] = model.observation(a.x[:,Int(time_point[_l]/dt) + 1], a.r)[1] + rand(d[1])
            # obs[N+1, Int(time_point[_l]/dt) + 1, _replicate] = exp(a.p[_l+1]) + rand(d[N+1])
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
        "--number-of-params"
            help = "#parameters"
            arg_type = Int
            default = 9
        "--number-of-obs-params"
            help = "number of observation parameters"
            arg_type = Int
            default = 4
        "--true-params", "-p"
            help = "true parameters"
            arg_type = Float64
            nargs = '*'
            # default = log.([0.6, 8., 0.97, 0.02, 0.02, 0.4, 1.0, 0.3, 0.01])
            default = log.([0.2, 0.1, 0.6, 0.02, 0.025, 0.5, 1.25, 0.7, 0.01, e, e, e, e])
            # default = log.([0.2, 0.1, 0.6, 0.02, 0.025, 0.05])
        "--initial-lower-bounds", "-l"
            help = "lower bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            # default = log.([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1., 0.5, 0.01, 0.01, 0.1, 1., 0.1, 0.005])
            default = log.([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.01, 0.01, 0.1, 1., 0.1, 0.005, 2., 2., 2., 2.])
            # default = log.([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.01, 0.01, 0.01])
        "--initial-upper-bounds", "-u"
            help = "upper bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            # default = log.([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 10., 2., 0.05, 0.05, 1., 5., 0.5, 0.02])
            default = log.([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.5, 1., 0.05, 0.05, 1., 5., 1., 0.02, 3., 3., 3., 3.])
            # default = log.([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.5, 1., 0.05, 0.05, 0.1])
        "--pseudo-obs"
            help = "#pseudo observations"
            arg_type = Int
            nargs = '+'
            default = [0, 0, 0]
        "--pseudo-obs-var"
            help = "variance of pseudo observations"
            arg_type = Float64
            nargs = '+'
            default = [1., 1., 1.]
        "--obs-variance"
            help = "observation variance"
            arg_type = Float64
            nargs = '+'
            # default = [0.1, 0.01]
            default = [0.0001, 0.0001, 0.0001]
        "--obs-iteration"
            help = "observation iteration"
            arg_type = Int
            # default = 15
            default = 20
            # default = 1
        "--dt"
            help = "Δt"
            arg_type = Float64
            # default = 0.025
            default = 0.25
        "--spinup"
            help = "spinup"
            arg_type = Float64
            default = 0.
        "--duration", "-t"
            help = "assimilation duration"
            arg_type = Float64
            default = 35.
            # default = 1.
        "--generation-seed", "-s"
            help = "seed for orbit generation"
            arg_type = Int
            default = 0
        "--trials"
            help = "#trials for gradient descent initial value"
            arg_type = Int
            # default = 100
            default = 10
        "--newton-maxiter"
            help = "#maxiter for newton's method"
            arg_type = Int
            default = 1000
        "--newton-tol"
            help = "newton method toralence"
            arg_type = Float64
            default = 1e-12
            # default = 1e-4
        "--regularization-coefficient"
            help = "regularization coefficient"
            arg_type = Float64
            default = 1.
        "--replicates"
            help = "#replicates"
            arg_type = Int
            # default = 1
            default = 10
        "--iter"
            help = "#iterations"
            arg_type = Int
            default = 1
        "--time-point"
            help = "time points"
            arg_type = Float64
            nargs = '+'
            # default = [0., 10., 20., 30., 40.]
            default = [0., 5., 10., 20., 35.]
            # default = [0., 1.]
        "--parameters"
            help = "name of the parameters"
            arg_type = String
            nargs = '+'
            default = ["log[STAT]", "log[pSTAT]", "log[pSTAT-pSTAT]", "log[npSTAT-npSTAT]", "log[nSTAT1]", "log[nSTAT2]", "log[nSTAT3]", "log[nSTAT4]", "log[nSTAT5]", "logp1", "logp2", "logp3", "logp4", "logu1", "logu2", "logu3", "logu4", "logu5", "Opstat", "Spstat", "Otstat", "Ststat"]
    end
    parsed_args = parse_args(args, settings; as_symbols=true) # ARGS is needed for static compilation; Otherwise, global ARGS is used.
    check_args(settings; parsed_args...)
    twin_experiment_obs_m!(dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, calc_eqdot!, observation!, observation_jacobianx!, observation_jacobianr!, observation_jacobianp!, observation_hessianxx!, observation_hessianxr!, observation_hessianrr!, observation_hessianpp!; parsed_args...)

    return 0
end

end
