using CatViews, Distributions, LineSearches, Optim, ArgParse, Gadfly, LinearAlgebra, Random, DelimitedFiles, Cairo, Fontconfig

include("../../src/types_backward_obsparams3.jl")
include("../../src/adjoint_backward_obsparam_lotka.jl")
include("../../src/assimilate_backward_obsparam2.jl")

include("../../util/check_args.jl")
include("../../util/check_gradient_backward_spline_attenuation_obsparams2.jl")
include("model2.jl")

zerodiv2zero(a,b) = (b==0.) ? 0. : a/b

@views function gradient_covariance_check_obs_m!(
    dxdt!::Function,
    jacobianx!::Function,
    jacobianp!::Function,
    hessianxx!::Function,
    hessianxp!::Function,
    hessianpp!::Function,
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
    numerical_differentiation_delta = nothing,
    time_point = nothing,
    parameters = nothing
    )

    obs_variance_bak = copy(obs_variance)

    N = length(initial_lower_bounds) - number_of_params - number_of_obs_params
    L = N + number_of_params
    R = number_of_obs_params
    U = length(obs_variance)
    println(N, L, R, U)
    model = Model(typeof(dt), N, L, R, U, time_point, dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, observation!, observation_jacobianx!, observation_jacobianr!, observation_jacobianp!, observation_hessianxx!, observation_hessianxr!, observation_hessianrr!, observation_hessianpp!)
    Random.seed!(generation_seed)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L+R]
    x0 = rand.(dists[1:N])
    a = Adjoint(dt, duration, number_of_obs_params, pseudo_obs, pseudo_obs_var, x0, copy(true_params), replicates, newton_maxiter, newton_tol, regularization_coefficient)
    orbit_first!(a, model)
    tob = deepcopy(a.x)
    dir *= "/true_params_$(join_digits3(true_params))/initial_lower_bounds_$(join_digits3(initial_lower_bounds))/initial_upper_bounds_$(join_digits3(initial_upper_bounds))/pseudo_obs_$(join_digits3(pseudo_obs))/pseudo_obs_var_$(join_digits3(pseudo_obs_var))/obs_variance_$(join_digits3(obs_variance))/spinup_$(digits3(spinup))/trials_$(digits3(trials))/newton_maxiter_$(digits3(newton_maxiter))/newton_tol_$(digits3(newton_tol))/regularization_coefficient_$(digits3(regularization_coefficient))/obs_iteration_$(digits3(obs_iteration))/dt_$(digits3(dt))/duration_$(digits3(duration))/replicates_$(digits3(replicates))/iter_$(digits3(iter))/numerical_differentiation_delta_$(digits10(numerical_differentiation_delta))/time_point_$(join_digits3(time_point))/"
    Random.seed!(hash([true_params, obs_variance_bak, obs_iteration, spinup, duration, generation_seed, replicates, iter, time_point]))

    d = Normal.(0., sqrt.(obs_variance_bak))
    obs = Array{typeof(dt)}(undef, U, a.steps+1, replicates)

    obs[:, :, :] .= NaN
    for _replicate in 1:replicates
        for _l in 1:length(time_point)
            observation!(model, time_point[_l], a.x[:,Int(time_point[_l]/dt) + 1], a.p)
            obs[:, Int(time_point[_l]/dt) + 1, _replicate] .= model.observation .+ rand.(d)
        end
    end

    obs_mean_var!(a, model, obs)

    println(dir)
    mkpath(dir)
    # plot_orbit(dir, a, model, a.x, obs)

    # θ = rand.(dists)
    θ = vcat(deepcopy(a.x[:,1]), copy(a.p))
    initialize!(a, θ)
    gr_ana = Vector{typeof(dt)}(undef,L+R)
    orbit_gradient!(a, model, gr_ana)
    gr_num = numerical_gradient!(a, model, numerical_differentiation_delta)
    println("analytical gradient:\t")
    Base.print_matrix(stdout, gr_ana')
    writedlm(dir * "analytical_gradient.tsv", gr_ana')
    println("numerical gradient:\t")
    Base.print_matrix(stdout, gr_num')
    writedlm(dir * "numerical_gradient.tsv", gr_num')
    diff = gr_ana .- gr_num
    println("absolute difference:\t")
    Base.print_matrix(stdout, diff')
    writedlm(dir * "gradient_absolute_difference.tsv", diff')
    println("max absolute difference:\t", maximum(abs, diff))
    if !any(gr_num == 0)
        rel_diff = diff ./ gr_num
        println("relative difference:\t")
        Base.print_matrix(stdout, rel_diff')
        writedlm(dir * "gradient_relative_difference.tsv", rel_diff')
        println("max relative difference:\t", maximum(abs, rel_diff))
    end

    white_panel = Theme(panel_fill="white")
    # p_stack = Array{Gadfly.Plot}(undef, 0)
    # p_stack = vcat(p_stack,
    # Gadfly.plot(
    # layer(x=tob[1:1,1], y=tob[2:2,1], Geom.point),
    # layer(z=(x,y) -> (initialize!(a, vcat([x], [y], tob[3:end,1], true_params)); orbit_cost!(a, model)), x=linspace(tob[1,1] - 0.1abs(tob[1,1]), tob[1,1] + 0.1abs(tob[1,1]), 100), y=linspace(tob[2,1] - 0.1abs(tob[2,1]), tob[2,1] + 0.1abs(tob[2,1]), 100), Geom.contour(levels=100)),
    # white_panel))
    # draw(PDF(dir * "cost_contour.pdf", 24cm, 24cm), vstack(p_stack))

    # p_stack = Array{Gadfly.Plot}(undef, 0)
    # p_stack = vcat(p_stack,
    # Gadfly.plot(
    # layer(x=tob[3:3,1], y=tob[4:4,1], Geom.point),
    # layer(z=(x,y) -> (initialize!(a, vcat(tob[1:2,1], [x,y], tob[5:end,1], true_params)); orbit_cost!(a, model)), x=linspace(initial_lower_bounds[3], 1., 100), y=linspace(initial_lower_bounds[4], 1., 100), Geom.contour(levels=1000)),
    # white_panel, Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"))))
    # draw(PDF(dir * "cost_contour_3_4.pdf", 24cm, 24cm), vstack(p_stack))
    #
    # p_stack = Array{Gadfly.Plot}(undef, 0)
    # p_stack = vcat(p_stack,
    # Gadfly.plot(
    # layer(x=tob[4:4,1], y=tob[5:5,1], Geom.point),
    # layer(z=(x,y) -> (initialize!(a, vcat(tob[1:3,1], [x,y], tob[6:end,1], true_params)); orbit_cost!(a, model)), x=linspace(initial_lower_bounds[4], 1., 100), y=linspace(initial_lower_bounds[5], 1., 100), Geom.contour(levels=1000)),
    # white_panel, Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"))))
    # draw(PDF(dir * "cost_contour_4_5.pdf", 24cm, 24cm), vstack(p_stack))
    #
    # p_stack = Array{Gadfly.Plot}(undef, 0)
    # p_stack = vcat(p_stack,
    # Gadfly.plot(
    # layer(x=tob[5:5,1], y=tob[6:6,1], Geom.point),
    # layer(z=(x,y) -> (initialize!(a, vcat(tob[1:4,1], [x,y], tob[7:end,1], true_params)); orbit_cost!(a, model)), x=linspace(-10., 1., 100), y=linspace(tob[6,1] - 0.5abs(tob[6,1]), tob[6,1] + 0.5abs(tob[6,1]), 100), Geom.contour(levels=1000)),
    # white_panel, Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"))))
    # draw(PDF(dir * "cost_contour_5_6.pdf", 24cm, 24cm), vstack(p_stack))
    #
    # p_stack = Array{Gadfly.Plot}(undef, 0)
    # p_stack = vcat(p_stack,
    # Gadfly.plot(
    # layer(x=tob[7:7,1], y=tob[8:8,1], Geom.point),
    # layer(z=(x,y) -> (initialize!(a, vcat(tob[1:6,1], [x,y], tob[9:end,1], true_params)); orbit_cost!(a, model)), x=linspace(2initial_lower_bounds[7], 1., 100), y=linspace(2initial_lower_bounds[8], 1., 100), Geom.contour(levels=1000)),
    # white_panel, Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"))))
    # draw(PDF(dir * "cost_contour_7_8.pdf", 24cm, 24cm), vstack(p_stack))
    #
    # p_stack = Array{Gadfly.Plot}(undef, 0)
    # p_stack = vcat(p_stack,
    # Gadfly.plot(
    # layer(x=tob[8:8,1], y=tob[9:9,1], Geom.point),
    # layer(z=(x,y) -> (initialize!(a, vcat(tob[1:7,1], [x,y], true_params)); orbit_cost!(a, model)), x=linspace(2initial_lower_bounds[8], 1., 100), y=linspace(2initial_lower_bounds[9], 1., 100), Geom.contour(levels=1000)),
    # white_panel, Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"))))
    # draw(PDF(dir * "cost_contour_8_9.pdf", 24cm, 24cm), vstack(p_stack))
    #
    # p_stack = Array{Gadfly.Plot}(undef, 0)
    # p_stack = vcat(p_stack,
    # Gadfly.plot(
    # layer(x=true_params[1:1], y=true_params[2:2], Geom.point),
    # layer(z=(x,y) -> (initialize!(a, vcat(tob[1:N,1], [x,y], true_params[3:L-N+R])); orbit_cost!(a, model)), x=linspace(initial_lower_bounds[N+1], initial_upper_bounds[N+1], 100), y=linspace(initial_lower_bounds[N+2], initial_upper_bounds[N+2], 100), Geom.contour(levels=1000)),
    # white_panel, Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"))))
    # draw(PDF(dir * "cost_contour_$(N+1)_$(N+2).pdf", 24cm, 24cm), vstack(p_stack))
    #
    # p_stack = Array{Gadfly.Plot}(undef, 0)
    # p_stack = vcat(p_stack,
    # Gadfly.plot(
    # layer(x=true_params[3:3], y=true_params[4:4], Geom.point),
    # layer(z=(x,y) -> (initialize!(a, vcat(tob[1:N,1], true_params[1:2], [x,y], true_params[5:L-N+R])); orbit_cost!(a, model)), x=linspace(initial_lower_bounds[N+3], initial_upper_bounds[N+3], 100), y=linspace(initial_lower_bounds[N+4], initial_upper_bounds[N+4], 100), Geom.contour(levels=500)),
    # white_panel, Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"))))
    # draw(PDF(dir * "cost_contour_$(N+3)_$(N+4).pdf", 24cm, 24cm), vstack(p_stack))
    #
    # p_stack = Array{Gadfly.Plot}(undef, 0)
    # p_stack = vcat(p_stack,
    # Gadfly.plot(
    # layer(x=true_params[L-N+1:L-N+1], y=true_params[L-N+2:L-N+2], Geom.point),
    # layer(z=(x,y) -> (initialize!(a, vcat(tob[1:N,1], true_params[1:L-N], [x,y], true_params[L-N+3:L-N+R])); orbit_cost!(a, model)), x=linspace(initial_lower_bounds[L+1], initial_upper_bounds[L+1], 100), y=linspace(initial_lower_bounds[L+2], initial_upper_bounds[L+2], 100), Geom.contour(levels=1000)),
    # white_panel, Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"))))
    # draw(PDF(dir * "cost_contour_$(L+1)_$(L+2).pdf", 24cm, 24cm), vstack(p_stack))
    #
    # p_stack = Array{Gadfly.Plot}(undef, 0)
    # p_stack = vcat(p_stack,
    # Gadfly.plot(
    # layer(x=true_params[L-N+3:L-N+3], y=true_params[L-N+R:L-N+R], Geom.point),
    # layer(z=(x,y) -> (initialize!(a, vcat(tob[1:N,1], true_params[1:L-N+2], [x,y])); orbit_cost!(a, model)), x=linspace(initial_lower_bounds[L+3], initial_upper_bounds[L+3], 100), y=linspace(initial_lower_bounds[L+4], initial_upper_bounds[L+4], 100), Geom.contour(levels=1000)),
    # white_panel, Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"))))
    # draw(PDF(dir * "cost_contour_$(L+3)_$(L+4).pdf", 24cm, 24cm), vstack(p_stack))

    println("ans: ")
    Base.print_matrix(stdout, CatView(tob[:,1], true_params)')
    println("")

    res_ana, minres = assimilate!(a, model, initial_lower_bounds, initial_upper_bounds, dists, trials)

    # p_stack = Array{Gadfly.Plot}(undef, 0)
    # p_stack = vcat(p_stack,
    # Gadfly.plot(
    # layer(x=minres.minimizer[1:1], y=minres.minimizer[2:2], Geom.point),
    # layer(z=(x,y) -> (initialize!(a, vcat([x, y], minres.minimizer[3:end])); orbit_cost!(a, model)), x=linspace(minres.minimizer[1] - abs(minres.minimizer[1]), minres.minimizer[1] + abs(minres.minimizer[1]), 100), y=linspace(minres.minimizer[2] - abs(minres.minimizer[2]), minres.minimizer[2] + abs(minres.minimizer[2]), 100), Geom.contour(levels=100)),
    # white_panel))
    # draw(PDF(dir * "cost_contour_minimizer.pdf", 24cm, 24cm), vstack(p_stack))

    # p_stack = Array{Gadfly.Plot}(undef, 0)
    # p_stack = vcat(p_stack,
    # Gadfly.plot(
    # layer(x=minres.minimizer[5:5], y=minres.minimizer[6:6], Geom.point),
    # layer(z=(x,y) -> (initialize!(a, vcat(minres.minimizer[1:4], [x,y], minres.minimizer[7:end])); orbit_cost!(a, model)), x=linspace(minres.minimizer[5] - abs(minres.minimizer[5]), minres.minimizer[5] + abs(minres.minimizer[5]), 100), y=linspace(minres.minimizer[6] - abs(minres.minimizer[6]), minres.minimizer[6] + abs(minres.minimizer[6]), 100), Geom.contour(levels=100)),
    # white_panel))
    # draw(PDF(dir * "cost_contour_minimizer.pdf", 24cm, 24cm), vstack(p_stack))

    res_num = numerical_covariance!(a, model, numerical_differentiation_delta)
    write_twin_experiment_result(dir, res_ana, minres.minimum, true_params, tob)

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
    if (res_ana.precision != nothing) && (res_num.precision != nothing)
        prec_ana = res_ana.precision
        prec_num = res_num.precision
        println("eigen:")
        Base.print_matrix(stdout, eigvals(prec_ana))
        println("")
        println("analytical precision:")
        Base.print_matrix(stdout, prec_ana)
        println("")
        writedlm(dir * "analytical_precision.tsv", prec_ana)
        println("numerical precision:")
        Base.print_matrix(stdout, prec_num)
        println("")
        writedlm(dir * "numerical_precision.tsv", prec_num)
        diff = prec_ana .- prec_num
        # println("absolute difference:\t")
        # Base.print_matrix(stdout, diff)
        writedlm(dir * "precision_absolute_difference.tsv", diff)
        println("max absolute difference:\t", maximum(abs, diff))
        if !any(res_num.stddev == 0)
            # rel_diff = diff ./ prec_num
            # rel_diff = zerodiv2zero.(diff, prec_num)
            rel_diff = zerodiv2zero.(diff, prec_ana)
            # println("relative difference:\t")
            # Base.print_matrix(stdout, rel_diff)
            writedlm(dir * "precision_relative_difference.tsv", rel_diff)
            println("max_relative_difference:\t", maximum(abs, rel_diff))
        end
        max_ana = maximum(abs, prec_ana)
        max_num = maximum(abs, prec_num)
        max_rel = maximum(abs, rel_diff)
        pl = hstack(spy(prec_ana, Scale.x_discrete(; labels= i->parameters[i]), Scale.y_discrete(; labels= i->parameters[i]), Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"), maxvalue=max_ana, minvalue=-max_ana), Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey("Analytical Precision"), Theme(panel_fill="white")),
                    spy(prec_num, Scale.x_discrete(; labels= i->parameters[i]), Scale.y_discrete(; labels= i->parameters[i]), Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"), maxvalue=max_num, minvalue=-max_num), Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey("Numerical Precision"), Theme(panel_fill="white")),
                    spy(rel_diff, Scale.x_discrete(; labels= i->parameters[i]), Scale.y_discrete(; labels= i->parameters[i]), Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"), maxvalue=max_rel, minvalue=-max_rel), Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey("Relative Difference"), Theme(panel_fill="white")))
        draw(PDF(dir * "precision.pdf", 48cm, 24cm), pl)

        println("obs_variabce:")
        Base.print_matrix(stdout, res_ana.obs_variance)
        println("covariance diag:")
        Base.print_matrix(stdout, diag(inv(res_ana.precision)))
    end
    plot_twin_experiment_result_wo_errorbar_observation(dir, a, model, tob, obs, true_params)
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
        "--number-of-params"
            help = "#parameters"
            arg_type = Int
            default = 4
        "--number-of-obs-params"
            help = "number of observation parameters"
            arg_type = Int
            default = 1
        "--true-params", "-p"
            help = "true parameters"
            arg_type = Float64
            nargs = '*'
            default = log.([1., 1., 1., 1., ℯ])
        "--initial-lower-bounds", "-l"
            help = "lower bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = log.([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.])
        "--initial-upper-bounds", "-u"
            help = "upper bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = log.([1., 1., 1., 1., 1., 1., 3.])
        "--pseudo-obs"
            help = "#pseudo observations"
            arg_type = Int
            nargs = '+'
            default = [0]
        "--pseudo-obs-var"
            help = "variance of pseudo observations"
            arg_type = Float64
            nargs = '+'
            default = [1.]
        "--obs-variance"
            help = "observation variance"
            arg_type = Float64
            nargs = '+'
            default = [0.001]
        "--obs-iteration"
            help = "observation iteration"
            arg_type = Int
            default = 20
        "--dt"
            help = "Δt"
            arg_type = Float64
            default = 0.25
        "--spinup"
            help = "spinup"
            arg_type = Float64
            default = 0.
        "--duration", "-t"
            help = "assimilation duration"
            arg_type = Float64
            default = 35.
        "--generation-seed", "-s"
            help = "seed for orbit generation"
            arg_type = Int
            default = 0
        "--trials"
            help = "#trials for gradient descent initial value"
            arg_type = Int
            default = 10
        "--newton-maxiter"
            help = "#maxiter for newton's method"
            arg_type = Int
            default = 1000
        "--newton-tol"
            help = "newton method toralence"
            arg_type = Float64
            default = 1e-15
        "--regularization-coefficient"
            help = "regularization coefficient"
            arg_type = Float64
            default = 0.01
        "--replicates"
            help = "#replicates"
            arg_type = Int
            default = 1000
        "--iter"
            help = "#iterations"
            arg_type = Int
            default = 1
        "--numerical-differentiation-delta"
            help = "numerical differentiation delta"
            arg_type = Float64
            default = 0.00000001
        "--time-point"
            help = "time points"
            arg_type = Float64
            nargs = '+'
            default = [0., 1., 3., 6., 8.]
        "--parameters"
            help = "name of the parameters"
            arg_type = String
            nargs = '+'
            default = ["logX0", "logY0", "logp1", "logp2", "logp3", "logp4", "O"]
    end

    parsed_args = parse_args(args, settings; as_symbols=true) # ARGS is needed for static compilation; Otherwise, global ARGS is used.
    check_args(settings; parsed_args...)

    gradient_covariance_check_obs_m!(dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, observation!, observation_jacobianx!, observation_jacobianr!, observation_jacobianp!, observation_hessianxx!, observation_hessianxr!, observation_hessianrr!, observation_hessianpp!; parsed_args...)

    return 0
end

julia_main(ARGS)
