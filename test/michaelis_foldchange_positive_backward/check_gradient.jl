include("../../src/AdjointsGivenZeroStateBackward.jl")

module Michaelis

using ArgParse, AdjointsGivenZeroStateBackward, Distributions, CatViews.CatView, Gadfly, Juno

export julia_main

include("../../util/optparser_given_zero_state.jl")
include("../../util/check_gradient_given_zero_state_backward.jl")
include("model.jl")

ct(x) = exp(-x)

function plot_twin_experiment_result_wo_errorbar_ct(dir, a::Adjoint{N,L,K}, m::Model, tob, obs, ct) where {N,L,K}
    white_panel = Theme(panel_fill="white")
    p_stack = Array{Gadfly.Plot}(0)
    t = collect(0.:a.dt:a.steps*a.dt)
    for _i in 1:N
        df_tob = DataFrame(t=t, x=ct.(tob[_i,:]), data_type="true orbit")
        # _mask = isfinite.(a.obs[_i,:,1])
        _mask = isfinite.(view(reshape(obs, N, :), _i, :))
        # df_obs = DataFrame(t=t[_mask], x=a.obs[_i,:,1][_mask], data_type="observed") # plot one replicate for example
        df_obs = DataFrame(t=view(repeat(t; outer=[K]), :)[_mask], x=ct.(m.inv_observation.(view(reshape(obs, N, :), _i, :)[_mask])), data_type="observed")
        df_assim = DataFrame(t=t, x=ct.(a.x[_i,:]), data_type="assimilated")
        p_stack = vcat(p_stack,
        Gadfly.plot(
        layer(df_tob, x="t", y="x", color=:data_type, Geom.line),
        layer(df_assim, x="t", y="x", color=:data_type, Geom.line),
        layer(df_obs, x="t", y="x", color=:data_type, Geom.point),
        Guide.xlabel("<i>t</i>"),
        Guide.ylabel("<i>x<sub>$_i</sub></i>"),
        white_panel))
    end
    draw(PDF(dir * "assimilation.pdf", 24cm, 40cm), vstack(p_stack))
    nothing
    # set_default_plot_size(24cm, 40cm)
    # vstack(p_stack)
end

function plot_orbit_ct(dir, a::Adjoint{N,L,K}, m::Model, tob, obs, ct) where {N,L,K}
    white_panel = Theme(panel_fill="white")
    p_stack = Array{Gadfly.Plot}(0)
    t = collect(0.:a.dt:a.steps*a.dt)
    for _i in 1:N
        df_tob = DataFrame(t=t, x=ct.(tob[_i,:]), data_type="true orbit")
        # _mask = isfinite.(a.obs[_i,:,1])
        _mask = isfinite.(view(reshape(obs, N, :), _i, :))
        # df_obs = DataFrame(t=t[_mask], x=a.obs[_i,:,1][_mask], data_type="observed") # plot one replicate for example
        df_obs = DataFrame(t=view(repeat(t; outer=[K]), :)[_mask], x=ct.(m.inv_observation.(view(reshape(obs, N, :), _i, :)[_mask])), data_type="observed")
        p_stack = vcat(p_stack,
        Gadfly.plot(
        layer(df_tob, x="t", y="x", color=:data_type, Geom.line),
        layer(df_obs, x="t", y="x", color=:data_type, Geom.point),
        Guide.xlabel("<i>t</i>"),
        Guide.ylabel("<i>x<sub>$_i</sub></i>"),
        white_panel))
    end
    draw(PDF(dir * "true_orbit_ct.pdf", 24cm, 40cm), vstack(p_stack))
    nothing
    # set_default_plot_size(24cm, 40cm)
    # vstack(p_stack)
end


@views function gradient_covariance_check_obs_m!(
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
    generation_seed = nothing,
    trials = nothing,
    replicates = nothing,
    iter = nothing,
    numerical_differentiation_delta = nothing,
    x0 = nothing,
    trials_nlsolver = nothing,
    search_box_nlsolver = nothing
    )

    obs_variance_bak = copy(obs_variance)

    N = length(x0)
    L = N + length(true_params)
    model = Model(typeof(dt), N, L, dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, observation, d_observation, dd_observation, inv_observation)
    srand(generation_seed)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L-N]
    a = Adjoint(dt, duration, pseudo_obs, pseudo_obs_var, x0, copy(true_params), replicates, trials_nlsolver, search_box_nlsolver)
    orbit!(a, model)
    tob = deepcopy(a.x)
    dir *= "/true_params_$(join(true_params, "_"))/initial_lower_bounds_$(join(initial_lower_bounds, "_"))/initial_upper_bounds_$(join(initial_upper_bounds, "_"))/pseudo_obs_$(join(pseudo_obs, "_"))/pseudo_obs_var_$(join(pseudo_obs_var, "_"))/spinup_$spinup/generation_seed_$generation_seed/trials_$trials/obs_variance_$(join(obs_variance_bak, "_"))/obs_iteration_$obs_iteration/dt_$dt/duration_$duration/replicates_$replicates/iter_$iter/"

    # srand(hash([true_params, initial_lower_bounds, initial_upper_bounds, pseudo_obs, pseudo_obs_var, obs_variance_bak, obs_iteration, dt, spinup, duration, generation_seed, trials, replicates, iter]))
    srand(hash([true_params, obs_variance_bak, obs_iteration, spinup, duration, generation_seed, replicates, iter]))

    d = Normal.(0., sqrt.(obs_variance_bak))
    obs = Array{typeof(dt)}(N, a.steps+1, replicates)

    obs[1,:,:] .= NaN
    for _replicate in 1:replicates
        obs[2,:,_replicate] .= model.observation.(a.x[2,:]) .+ rand(d[2], a.steps+1)
        for _i in 1:obs_iteration:a.steps
            for _k in 1:obs_iteration-1
                obs[2, _i + _k, _replicate] = NaN
            end
        end
    end
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
    obs[:,1,:] .= NaN
    obs_mean_var!(a, model, obs)

    println(dir)
    mkpath(dir)
    # plot_orbit(dir, a, model, a.x, obs)
    # plot_orbit_ct(dir, a, model, a.x, obs, ct)

    p = rand.(dists)
    initialize_p!(a, p)
    gr_ana = Vector{typeof(dt)}(L-N)
    orbit_gradient!(a, model, gr_ana)
    gr_num = numerical_gradient!(a, model, numerical_differentiation_delta)
    println("analytical gradient:\t", gr_ana)
    println("numerical gradient:\t", gr_num)
    diff = gr_ana .- gr_num
    println("absolute difference:\t", diff)
    println("max absolute difference:\t", maximum(abs, diff))
    if !any(gr_num == 0)
        rel_diff = diff ./ gr_num
        println("relative difference:\t", rel_diff)
        println("max relative difference:\t", maximum(abs, rel_diff))
    end

    res_ana, minres = assimilate!(a, model, initial_lower_bounds, initial_upper_bounds, dists, trials)

    res_num = numerical_covariance!(a, model, numerical_differentiation_delta)
    write_twin_experiment_result(dir, res_ana, minres.minimum, true_params, tob)

    white_panel = Theme(panel_fill="white")
    p_stack = Array{Gadfly.Plot}(0)
    # p_stack = vcat(p_stack,
    # Gadfly.plot(
    # layer(x=minres.minimizer[1:1], y=minres.minimizer[2:2], Geom.point),
    # layer(z=(x,y) -> (initialize_p!(a, [x, y, minres.minimizer[3], minres.minimizer[4]]); orbit_negative_log_likelihood!(a, model, pseudo_obs, pseudo_obs_var)), x=linspace(1e-5, max(initial_upper_bounds[1], 2.*minres.minimizer[1]), 100), y=linspace(1e-5, max(initial_upper_bounds[2], 2.*minres.minimizer[2]), 100), Geom.contour(levels=100)),
    # white_panel))

    # p_stack = vcat(p_stack,
    # Gadfly.plot(
    # layer(x=minres.minimizer[1:1], y=minres.minimizer[2:2], Geom.point),
    # layer(z=(x,y) -> (initialize_p!(a, [x, y, minres.minimizer[3], minres.minimizer[4]]); orbit_cost!(a, model)), x=linspace(1e-5, max(initial_upper_bounds[1], 10.*minres.minimizer[1]), 100), y=linspace(1e-5, max(initial_upper_bounds[2], 10.*minres.minimizer[2]), 100), Geom.contour(levels=500)),
    # white_panel))
    # p_stack = vcat(p_stack,
    # Gadfly.plot(
    # layer(x=minres.minimizer[3:3], y=minres.minimizer[4:4], Geom.point),
    # layer(z=(x,y) -> (initialize_p!(a, [minres.minimizer[1], minres.minimizer[2], x, y]); orbit_cost!(a, model)), x=linspace(1e-5, max(initial_upper_bounds[3], 10.*minres.minimizer[3]), 100), y=linspace(1e-5, max(initial_upper_bounds[4], 10.*minres.minimizer[4]), 100), Geom.contour(levels=500)),
    # white_panel))
    # draw(PDF(dir * "cost_contour.pdf", 24cm, 48cm), vstack(p_stack))

    initialize_p!(a, minres.minimizer)
    orbit_cost!(a, model)

    # if !isnull(res_ana.stddev) && !isnull(res_num.stddev)
    #     cov_ana = get(res_ana.stddev)
    #     cov_num = get(res_num.stddev)
    #     println("analytical stddev:\t", cov_ana)
    #     println("numerical stddev:\t", cov_num)
    #     diff = cov_ana .- cov_num
    #     println("absolute difference:\t", diff)
    #     println("max absolute difference:\t", maximum(abs, diff))
    #     if !any(res_num.stddev == 0)
    #         rel_diff = diff ./ cov_num
    #         println("relative difference:\t", rel_diff)
    #         println("max_relative_difference:\t", maximum(abs, rel_diff))
    #     end
    # end
    if !isnull(res_ana.precision) && !isnull(res_num.precision)
        prec_ana = get(res_ana.precision)
        prec_num = get(res_num.precision)
        println("analytical precision:\t", prec_ana)
        println("numerical precision:\t", prec_num)
        diff = prec_ana .- prec_num
        println("absolute difference:\t", diff)
        println("max absolute difference:\t", maximum(abs, diff))
        if !any(res_num.stddev == 0)
            rel_diff = diff ./ prec_num
            println("relative difference:\t", rel_diff)
            println("max_relative_difference:\t", maximum(abs, rel_diff))
        end
    end
    plot_twin_experiment_result_wo_errorbar_ct(dir, a, model, tob, obs, ct)
    plot_twin_experiment_result_wo_errorbar_observation(dir, a, model, tob, obs)
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
            default = "check"
        "--true-params", "-p"
            help = "true parameters"
            arg_type = Float64
            nargs = '*'
            default = [0.3, 1., 5., 0.15]
            # default = [0.3 + 1.4857221997930168, 1. - 0.533343425791887, 5. + 1.3203485637379382, 0.15 + 150.7318906200313]
        "--initial-lower-bounds", "-l"
            help = "lower bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = [0., 0., 0., 0.]
        "--initial-upper-bounds", "-u"
            help = "upper bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = [1., 10., 10., 1.]
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
        "--obs-variance"
            help = "observation variance"
            arg_type = Float64
            nargs = '+'
            # default = [0.1, 0.01]
            default = [1., 0.001]
        "--obs-iteration"
            help = "observation iteration"
            arg_type = Int
            # default = 5
            default = 15
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
        "--generation-seed", "-s"
            help = "seed for orbit generation"
            arg_type = Int
            default = 0
        "--trials"
            help = "#trials for gradient descent initial value"
            arg_type = Int
            # default = 100
            default = 100
        "--replicates"
            help = "#replicates"
            arg_type = Int
            # default = 1
            default = 100
        "--iter"
            help = "#iterations"
            arg_type = Int
            default = 1
        "--numerical-differentiation-delta"
            help = "numerical differentiation delta"
            arg_type = Float64
            default = 0.000000001
        "--x0"
            help = "initial x"
            arg_type = Float64
            nargs = '+'
            default = [0., 0.]
        "--trials-nlsolver"
            help = "#trials for NLSolve"
            arg_type = Int
            default = 100
        "--search-box-nlsolver"
            help = "size of nlsolver search boundingbox centerd on x_prev"
            arg_type = Float64
            nargs = '+'
            default = [1., 1.]
    end

    parsed_args = parse_args(args, settings; as_symbols=true) # ARGS is needed for static compilation; Otherwise, global ARGS is used.
    check_args(settings; parsed_args...)

    gradient_covariance_check_obs_m!(dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, observation, d_observation, dd_observation, inv_observation; parsed_args...)

    return 0
end

end
