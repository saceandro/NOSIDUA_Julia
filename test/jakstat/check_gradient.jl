using CatViews.CatView, Distributions, LineSearches, Optim, ArgParse, Gadfly, Juno

include("../../src/types_backward.jl")
include("../../src/adjoint_backward.jl")
include("../../src/assimilate_backward.jl")

include("../../util/check_args.jl")
include("../../util/check_gradient_backward.jl")
include("model.jl")

@views function gradient_covariance_check_obs_m!(
    dxdt!::Function,
    jacobianx!::Function,
    jacobianp!::Function,
    hessianxx!::Function,
    hessianxp!::Function,
    hessianpp!::Function,
    calc_qdot!::Function,
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
    newton_maxiter = nothing,
    newton_tol = nothing,
    regularization_coefficient = nothing,
    replicates = nothing,
    iter = nothing,
    numerical_differentiation_delta = nothing,
    time_point = nothing,
    parameters = nothing
    )

    obs_variance_bak = copy(obs_variance)

    L = length(initial_lower_bounds)
    N = L - length(true_params)
    model = Model(typeof(dt), N, L, time_point, dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, calc_qdot!, observation, d_observation, dd_observation, inv_observation)
    srand(generation_seed)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L]
    x0 = rand.(dists[1:N])
    a = Adjoint(dt, duration, pseudo_obs, pseudo_obs_var, x0, copy(true_params), replicates, newton_maxiter, newton_tol, regularization_coefficient)
    orbit!(a, model)
    tob = deepcopy(a.x)
    dir *= "/true_params2/"
    # dir *= "/true_params_$(join(true_params, "_"))/initial_lower_bounds_$(join(initial_lower_bounds, "_"))/initial_upper_bounds_$(join(initial_upper_bounds, "_"))/pseudo_obs_$(join(pseudo_obs, "_"))/pseudo_obs_var_$(join(pseudo_obs_var, "_"))/spinup_$spinup/trials_$trials/newton_maxiter_$newton_maxiter/obs_variance_$obs_variance/obs_iteration_$obs_iteration/dt_$dt/duration_$duration/replicates_$replicates/iter_$iter/"
    # srand(hash([true_params, initial_lower_bounds, initial_upper_bounds, pseudo_obs, pseudo_obs_var, obs_variance_bak, obs_iteration, dt, spinup, duration, generation_seed, trials, replicates, iter]))
    srand(hash([true_params, obs_variance_bak, obs_iteration, spinup, duration, generation_seed, replicates, iter]))

    d = Normal.(0., sqrt.(obs_variance_bak))
    obs = Array{typeof(dt)}(N, a.steps+1, replicates)

    for _j in 1:N
        for _replicate in 1:replicates
            obs[_j,:,_replicate] .= model.observation.(a.x[_j,:]) .+ rand(d[_j], a.steps+1)
            for _i in 1:obs_iteration:a.steps
                for _k in 1:obs_iteration-1
                    obs[_j, _i + _k, _replicate] = NaN
                end
            end
        end
    end
    obs_mean_var!(a, model, obs)

    println(dir)
    mkpath(dir)
    plot_orbit(dir, a, model, a.x, obs)

    draw(PDF(dir * "control.pdf", 24cm, 24cm), plot(
                                                    # layer(t -> u(model, t, true_params[5:end]), time_point[1], time_point[end], Geom.line, Theme(default_color=color("white"))),
                                                    layer(t -> u2(model, t, true_params[5:end]), time_point[1], time_point[end], Geom.line, Theme(default_color=color("orange"))),
                                                    layer(x=time_point, y=exp.(true_params[5:end]), Geom.point, Theme(default_color=color("orange"))),
                                                    layer(t -> du2(model, t, true_params[5:end]), time_point[1], time_point[end], Geom.line, Theme(default_color=color("blue"))),
                                                    layer(t -> ddu2(model, t, true_params[5:end]), time_point[1], time_point[end], Geom.line, Theme(default_color=color("green"))), Theme(panel_fill="white")))

    # θ = rand.(dists)
    θ = vcat(deepcopy(a.x[:,1]), copy(a.p))
    initialize!(a, θ)
    gr_ana = Vector{typeof(dt)}(L)
    orbit_gradient!(a, model, gr_ana)
    gr_num = numerical_gradient!(a, model, numerical_differentiation_delta)
    println("analytical gradient:\t")
    Base.print_matrix(STDOUT, gr_ana')
    writedlm(dir * "analytical_gradient.tsv", gr_ana')
    println("numerical gradient:\t")
    Base.print_matrix(STDOUT, gr_num')
    writedlm(dir * "numarical_gradient.tsv", gr_num')
    diff = gr_ana .- gr_num
    println("absolute difference:\t")
    Base.print_matrix(STDOUT, diff')
    writedlm(dir * "gradient_absolute_difference.tsv", diff')
    println("max absolute difference:\t", maximum(abs, diff))
    if !any(gr_num == 0)
        rel_diff = diff ./ gr_num
        println("relative difference:\t")
        Base.print_matrix(STDOUT, rel_diff')
        writedlm(dir * "gradient_relative_difference.tsv", rel_diff')
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

    initialize!(a, minres.minimizer)
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
        println("analytical precision:\t")
        Base.print_matrix(STDOUT, prec_ana)
        writedlm(dir * "analytical_precision.tsv", prec_ana)
        println("numerical precision:\t")
        Base.print_matrix(STDOUT, prec_num)
        writedlm(dir * "numerical_precision.tsv", prec_num)
        diff = prec_ana .- prec_num
        println("absolute difference:\t")
        Base.print_matrix(STDOUT, diff)
        writedlm(dir * "precision_absolute_difference.tsv", diff)
        println("max absolute difference:\t", maximum(abs, diff))
        if !any(res_num.stddev == 0)
            rel_diff = diff ./ prec_num
            println("relative difference:\t")
            Base.print_matrix(STDOUT, rel_diff)
            writedlm(dir * "precision_relative_difference.tsv", rel_diff)
            println("max_relative_difference:\t", maximum(abs, rel_diff))
        end
        max_ana = maximum(abs, prec_ana)
        max_num = maximum(abs, prec_num)
        max_rel = maximum(abs, rel_diff)
        pl = hstack(spy(prec_ana, Scale.x_discrete(; labels= i->parameters[i]), Scale.y_discrete(; labels= i->parameters[i]), Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"), maxvalue=max_ana, minvalue=-max_ana), Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey("Analytical Precision"), Theme(panel_fill="white")),
                    spy(prec_num, Scale.x_discrete(; labels= i->parameters[i]), Scale.y_discrete(; labels= i->parameters[i]), Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"), maxvalue=max_num, minvalue=-max_num), Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey("Numerical Precision"), Theme(panel_fill="white")),
                    spy(rel_diff, Scale.x_discrete(; labels= i->parameters[i]), Scale.y_discrete(; labels= i->parameters[i]), Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"), maxvalue=max_rel, minvalue=-max_rel), Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey("Relative Difference"), Theme(panel_fill="white")))
        draw(PDF(dir * "precision_dt0.025.pdf", 48cm, 24cm), pl)
    end
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
            default = "check3"
        "--true-params", "-p"
            help = "true parameters"
            arg_type = Float64
            nargs = '*'
            # default = log.([0.6, 8., 0.97, 0.02, 0.02, 0.4, 1.0, 0.3, 0.01])
            default = log.([0.2, 0.1, 0.6, 0.02, 0.025, 0.5, 1.25, 0.7, 0.01])
            # default = log.([0.2, 0.1, 0.6, 0.02, 0.025, 0.05])
        "--initial-lower-bounds", "-l"
            help = "lower bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            # default = log.([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1., 0.5, 0.01, 0.01, 0.1, 1., 0.1, 0.005])
            default = log.([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.01, 0.01, 0.1, 1., 0.1, 0.005])
            # default = log.([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.01, 0.01, 0.01])
        "--initial-upper-bounds", "-u"
            help = "upper bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            # default = log.([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 10., 2., 0.05, 0.05, 1., 5., 0.5, 0.02])
            default = log.([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.5, 1., 0.05, 0.05, 1., 5., 1., 0.02])
            # default = log.([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.5, 1., 0.05, 0.05, 0.1])
        "--pseudo-obs"
            help = "#pseudo observations"
            arg_type = Int
            nargs = '+'
            default = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        "--pseudo-obs-var"
            help = "variance of pseudo observations"
            arg_type = Float64
            nargs = '+'
            default = [1., 1., 1., 1., 1., 1., 1., 1., 1.]
        "--obs-variance"
            help = "observation variance"
            arg_type = Float64
            nargs = '+'
            # default = [0.1, 0.01]
            default = [0.001, 0.001, 0.0001, 0.001, 0.001, 0.0001, 0.0001, 0.00001, 0.0001]
        "--obs-iteration"
            help = "observation iteration"
            arg_type = Int
            # default = 15
            default = 20
            # default = 1
        "--dt"
            help = "Δt"
            arg_type = Float64
            default = 0.025
            # default = 1.
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
            default = 200
        "--newton-tol"
            help = "newton method toralence"
            arg_type = Float64
            default = 1e-8
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
        "--numerical-differentiation-delta"
            help = "numerical differentiation delta"
            arg_type = Float64
            # default = 0.0000001
            default = 0.00000001
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
            default = ["[STAT]", "[pSTAT]", "[pSTAT-pSTAT]", "[npSTAT-npSTAT]", "[nSTAT1]", "[nSTAT2]", "[nSTAT3]", "[nSTAT4]", "[nSTAT5]", "p1", "p2", "p3", "p4", "u1", "u2", "u3", "u4", "u5"]
    end

    parsed_args = parse_args(args, settings; as_symbols=true) # ARGS is needed for static compilation; Otherwise, global ARGS is used.
    check_args(settings; parsed_args...)

    gradient_covariance_check_obs_m!(dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, calc_eqdot!, observation, d_observation, dd_observation, inv_observation; parsed_args...)

    return 0
end

julia_main(ARGS)
