include("../../src/AdjointsGivenZeroState.jl")

module Michaelis

using ArgParse, AdjointsGivenZeroState, Distributions, CatViews.CatView, DataFrames, CSV, Gadfly

export julia_main

include("../../util/check_args_given_zero_state.jl")
include("../../util/assimilation_given_zero_state.jl")
include("model.jl")

ct(x) = exp(-x)

function plot_twin_experiment_result_wo_errorbar_ct(dir, file, a::Adjoint{N,L,K}, m::Model, obs, ct) where {N,L,K}
    white_panel = Theme(panel_fill="white")
    p_stack = Array{Gadfly.Plot}(0)
    t = collect(0.:a.dt:a.steps*a.dt)
    for _i in 1:N
        _mask = isfinite.(view(reshape(obs, N, :), _i, :))
        df_obs = DataFrame(t=view(repeat(t; outer=[K]), :)[_mask], x=ct.(m.inv_observation.(view(reshape(obs, N, :), _i, :)[_mask])), data_type="observed")
        df_assim = DataFrame(t=t, x=ct.(a.x[_i,:]), data_type="assimilated")
        p_stack = vcat(p_stack,
        Gadfly.plot(
        layer(df_assim, x="t", y="x", color=:data_type, Geom.line),
        layer(df_obs, x="t", y="x", color=:data_type, Geom.point),
        Guide.xlabel("<i>t</i>"),
        Guide.ylabel("<i>x<sub>$_i</sub></i>"),
        white_panel))
    end
    draw(PDF(dir * "$(file)_assimilation.pdf", 24cm, 40cm), vstack(p_stack))
    nothing
    # set_default_plot_size(24cm, 40cm)
    # vstack(p_stack)
end

@views function assimilation_plot!(
    dxdt!::Function,
    jacobian!::Function,
    hessian!::Function,
    observation::Function,
    d_observation::Function,
    dd_observation::Function,
    inv_observation::Function;
    dir = nothing,
    file = nothing,
    initial_lower_bounds = nothing,
    initial_upper_bounds = nothing,
    pseudo_obs = nothing,
    pseudo_obs_var = nothing,
    dt = nothing,
    duration = nothing,
    trials = nothing,
    x0 = nothing
    )

    # assimilation!(dxdt!, jacobian!, hessian!, observation, d_observation, dd_observation, inv_observation; dir = dir, file = file, initial_lower_bounds = initial_lower_bounds, initial_upper_bounds = initial_upper_bounds, pseudo_obs = pseudo_obs, pseudo_obs_var = pseudo_obs_var, dt = dt, duration = duration, trials = trials, x0 = x0)
    N = length(x0)
    M = length(initial_lower_bounds)
    L = N + M
    model = Model(typeof(dt), N, L, dxdt!, jacobian!, hessian!, observation, d_observation, dd_observation, inv_observation)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:M]

    d = CSV.read("$dir/$file"; delim='\t')
    K = size(d, 1)
    steps = Int(duration/dt)
    obs = Array{typeof(dt)}(N, steps+1, K)
    fill!(obs, NaN)
    for item in names(d)
           obs[2,Int(parse(Int, string(item))/dt)+1,:] .= get.(d[:, item])
    end
    a = Adjoint(M, dt, obs, pseudo_obs, pseudo_obs_var, x0)
    dir *= "/initial_lower_bounds_$(join(initial_lower_bounds, "_"))/initial_upper_bounds_$(join(initial_upper_bounds, "_"))/pseudo_obs_$(join(pseudo_obs, "_"))/pseudo_obs_var_$(join(pseudo_obs_var, "_"))/trials_$trials/dt_$dt/duration_$duration/"
    srand(hash([initial_lower_bounds, initial_upper_bounds, pseudo_obs, pseudo_obs_var, dt, duration, trials]))
    assimilation_procedure!(dir, file, a, model, initial_lower_bounds, initial_upper_bounds, dists, trials)

    plot_twin_experiment_result_wo_errorbar_ct(dir, file, a, model, obs, ct)
    plot_twin_experiment_result_wo_errorbar_observation(dir, file, a, model, obs)
end

@views function assimilation!(
    dxdt!::Function,
    jacobian!::Function,
    hessian!::Function,
    observation::Function,
    d_observation::Function,
    dd_observation::Function,
    inv_observation::Function;
    dir = nothing,
    file = nothing,
    initial_lower_bounds = nothing,
    initial_upper_bounds = nothing,
    pseudo_obs = nothing,
    pseudo_obs_var = nothing,
    dt = nothing,
    duration = nothing,
    trials = nothing,
    x0 = nothing
    )

    N = length(x0)
    M = length(initial_lower_bounds)
    L = N + M
    model = Model(typeof(dt), N, L, dxdt!, jacobian!, hessian!, observation, d_observation, dd_observation, inv_observation)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:M]

    d = CSV.read("$dir/$file"; delim='\t')
    K = size(d, 1)
    steps = Int(duration/dt)
    obs = Array{typeof(dt)}(N, steps+1, K)
    fill!(obs, NaN)
    for item in names(d)
           obs[2,parse(Int, string(item))+1,:] .= get.(d[:, item])
    end
    a = Adjoint(M, dt, obs, pseudo_obs, pseudo_obs_var, x0)
    dir *= "/initial_lower_bounds_$(join(initial_lower_bounds, "_"))/initial_upper_bounds_$(join(initial_upper_bounds, "_"))/pseudo_obs_$(join(pseudo_obs, "_"))/pseudo_obs_var_$(join(pseudo_obs_var, "_"))/trials_$trials/dt_$dt/duration_$duration/"
    srand(hash([initial_lower_bounds, initial_upper_bounds, pseudo_obs, pseudo_obs_var, dt, duration, trials]))
    assimilation_procedure!(dir, file, a, model, initial_lower_bounds, initial_upper_bounds, dists, trials)
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
            default = "data"
        "--file", "-f"
            help = "input data"
            default = "hmgcr.txt"
        # "--true-params", "-p"
        #     help = "true parameters"
        #     arg_type = Float64
        #     nargs = '*'
        #     default = [0.5, 1., 5., 4.]
        "--initial-lower-bounds", "-l"
            help = "lower bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = [0., 0., 0., 0.]
        "--initial-upper-bounds", "-u"
            help = "upper bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = [10., 10., 10., 10.]
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
        # "--obs-iteration"
        #     help = "observation iteration"
        #     arg_type = Int
        #     default = 5
        "--dt"
            help = "Δt"
            arg_type = Float64
            default = 0.1
        # "--spinup"
        #     help = "spinup"
        #     arg_type = Float64
        #     default = 0.
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
            default = 100
        # "--replicates"
        #     help = "#replicates"
        #     arg_type = Int
        #     default = 1
        # "--iter"
        #     help = "#iterations"
        #     arg_type = Int
        #     default = 1
        "--x0"
            help = "initial x"
            arg_type = Float64
            nargs = '+'
            default = [0., 0.]
    end

    parsed_args = parse_args(args, settings; as_symbols=true) # ARGS is needed for static compilation; Otherwise, global ARGS is used.
    # check_args(settings; parsed_args...)
    assimilation_plot!(dxdt!, jacobian!, hessian!, observation, d_observation, dd_observation, inv_observation; parsed_args...)

    return 0
end

end
