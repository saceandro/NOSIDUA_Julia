using CatViews, Distributions, LineSearches, Optim, ArgParse, Gadfly, LinearAlgebra, Random, DelimitedFiles, Cairo, Fontconfig, ASTInterpreter2, Juno

include("../../src/types_stochastic.jl")
include("../../src/adjoint_stochastic.jl")
include("../../src/assimilate_backward_obsparam2.jl")

include("../../util/check_args.jl")
# include("../../util/check_gradient_backward_spline_attenuation_obsparams2.jl")
include("model.jl")

zerodiv2zero(a,b) = (b==0.) ? 0. : a/b

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

function plot_orbit(dir, a::Adjoint{N,L,R,U,K}, m::Model, tob, obs, max_prob_path) where {N,L,R,U,K}
    white_panel = Theme(panel_fill="white")
    p_stack = Array{Gadfly.Plot}(undef, 0)
    t = collect(0.:a.dt:a.steps*a.dt)
    df_tob = DataFrame(t=t, x=[(m.observation!(m, t[_j], tob[:,_j], a.p); m.observation[1]) for _j in 1:a.steps+1], data_type="true orbit")
    df_tob = DataFrame(t=t, x=tob[1,:], data_type="true orbit")
    # df_tob = DataFrame(t=t, x=m.observation.(tob, a.r), data_type="true orbit")
    # _mask = isfinite.(a.obs[_i,:,1])
    _mask = isfinite.(view(reshape(obs, U, :), 1, :))
    # df_obs = DataFrame(t=t[_mask], x=a.obs[_i,:,1][_mask], data_type="observed") # plot one replicate for example
    df_obs = DataFrame(t=view(repeat(t; outer=[K]), :)[_mask], x=view(reshape(obs, U, :), 1, :)[_mask], data_type="observed")
    df_max_prob = DataFrame(t=view(repeat(t; outer=[K]), :)[_mask], x=view(reshape(max_prob_path, U, :), 1, :)[_mask], data_type="max prob path")
    p_stack = vcat(p_stack,
    Gadfly.plot(
    layer(df_tob, x="t", y="x", color=:data_type, Geom.line, Theme(default_color=color("blue"))),
    layer(df_max_prob, x="t", y="x", color=:data_type, Geom.line, Theme(default_color=color("green"))),
    layer(df_obs, x="t", y="x", color=:data_type, Geom.point, Theme(default_color=color("orange"))),
    Guide.xlabel("<i>t</i>"),
    Guide.ylabel("<i>x</i>"),
    white_panel))
    draw(PDF(dir * "true_orbit.pdf", 24cm, 24cm), vstack(p_stack))
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
    observation!::Function,
    observation_jacobianx!::Function,
    observation_jacobianr!::Function,
    observation_hessianxx!::Function,
    observation_hessianxr!::Function,
    observation_hessianrr!::Function;
    dir = nothing,
    true_params = nothing,
    true_obs_params = nothing,
    lower_bounds = nothing,
    upper_bounds = nothing,
    lower_bounds_params = nothing,
    upper_bounds_params = nothing,
    lower_bounds_obs_params = nothing,
    upper_bounds_obs_params = nothing,
    system_noise = nothing,
    pseudo_obs = nothing,
    pseudo_obs_var = nothing,
    obs_variance = nothing,
    obs_iteration = nothing,
    dt = nothing,
    spinup = nothing,
    duration = nothing,
    generation_seed = nothing,
    trials = nothing,
    regularization_coefficient = nothing,
    replicates = nothing,
    iter = nothing,
    numerical_differentiation_delta = nothing,
    time_point = nothing,
    parameters = nothing,
    obs_parameters = nothing
    )

    xdim = length(lower_bounds)
    pdim = length(lower_bounds_params)
    rdim = length(lower_bounds_obs_params)
    obsdim = length(obs_variance)
    model = Model(typeof(dt), xdim, pdim, rdim, obsdim, time_point, dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, observation!, observation_jacobianx!, observation_jacobianr!, observation_hessianxx!, observation_hessianxr!, observation_hessianrr!)

    Random.seed!(generation_seed)
    x0dists = [Uniform(lower_bounds[i], upper_bounds[i]) for i in 1:xdim]
    pdists  = [Uniform(lower_bounds_params[i], upper_bounds_params[i]) for i in 1:pdim]
    rdists  = [Uniform(lower_bounds_obs_params[i], upper_bounds_obs_params[i]) for i in 1:rdim]
    x0 = rand.(x0dists)

    # ASTInterpreter2.@enter Adjoint(dt, duration, pseudo_obs, pseudo_obs_var, x0, copy(true_params), copy(true_obs_params), replicates, regularization_coefficient)

    a = Adjoint(dt, duration, system_noise, obs_variance, pseudo_obs, pseudo_obs_var, x0, copy(true_params), copy(true_obs_params), replicates, regularization_coefficient)

    orbit_first!(a, model)
    tob = deepcopy(a.x)
    dir *= "/true_params_$(join_digits3(true_params))/true_obs_params_$(join_digits3(true_obs_params))/lower_bounds_$(join_digits3(lower_bounds))/upper_bounds_$(join_digits3(upper_bounds))/lower_bounds_params_$(join_digits3(lower_bounds_params))/upper_bounds_params_$(join_digits3(upper_bounds_params))/lower_bounds_obs_params_$(join_digits3(lower_bounds_obs_params))/upper_bounds_obs_params_$(join_digits3(upper_bounds_obs_params))/pseudo_obs_$(join_digits3(pseudo_obs))/pseudo_obs_var_$(join_digits3(pseudo_obs_var))/obs_variance_$(join_digits3(obs_variance))/spinup_$(digits3(spinup))/trials_$(digits3(trials))/regularization_coefficient_$(digits3(regularization_coefficient))/obs_iteration_$(digits3(obs_iteration))/dt_$(digits3(dt))/duration_$(digits3(duration))/replicates_$(digits3(replicates))/iter_$(digits3(iter))/numerical_differentiation_delta_$(digits10(numerical_differentiation_delta))/"
    Random.seed!(hash([true_params, true_obs_params, obs_variance, obs_iteration, spinup, duration, generation_seed, replicates, iter, time_point]))

    d = Normal.(0., sqrt.(obs_variance))
    obs = Array{typeof(dt)}(undef, obsdim, a.steps+1, replicates)

    obs[:, :, :] .= NaN
    for _replicate in 1:replicates
        for _l in 1:length(time_point)
            observation!(model, time_point[_l], a.x[:,Int(time_point[_l]/dt) + 1], a.r)
            obs[:, Int(time_point[_l]/dt) + 1, _replicate] .= model.observation .+ rand.(d)
        end
    end

    obs_mean_var!(a, model, obs)

    println("hoge")
    println(a.x[:,1])
    println(a.obs_mean[:,1])
    a.x[:,1] .= a.obs_mean[:,1]
    orbit!(a, model, a.obs_mean)

    println(dir)
    mkpath(dir)

    println("obs: $(a.obs_mean)")
    println("tob: $tob")
    println("a.x: $(a.x)")
    println("sys_precision: $(a.sys_precision)")
    println("obs_precision: $(a.obs_precision)")

    plot_orbit(dir, a, model, tob, obs, a.x)


    # θ = rand.(dists)
    θ = vcat(deepcopy(a.x[:,1]), copy(a.p), copy(a.r))
    initialize!(a, θ)
    gr_ana = Vector{typeof(dt)}(undef,xdim+pdim+rdim)
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
    println("ans: ")
    Base.print_matrix(stdout, CatView(tob[:,1], true_params, true_obs_params)')
    println("")

    res_ana, minres = assimilate!(a, model, lower_bounds, upper_bounds, lower_bounds_params, upper_bounds_params, lower_bounds_obs_params, upper_bounds_obs_params, dists, trials)

    res_num = numerical_covariance!(a, model, numerical_differentiation_delta)
    write_twin_experiment_result(dir, res_ana, minres.minimum, true_params, true_obs_params, tob)

    initialize!(a, minres.minimizer)
    orbit_cost!(a, model)

    if (res_ana.precision != nothing) && (res_num.precision != nothing)
        prec_ana = res_ana.precision
        prec_num = res_num.precision
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
    plot_twin_experiment_result_wo_errorbar_observation(dir, a, model, tob, obs, true_params, true_obs_params)
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
            arg_type = String
            default = "check"
        "--true-params", "-p"
            help = "true parameters"
            arg_type = Float64
            nargs = '+'
            default = [0.2]
        "--true-obs-params", "-r"
            help = "observation parameters"
            arg_type = Float64
            nargs = '*'
            default = [0.]
        "--lower-bounds", "-l"
            help = "lower bounds for initial state"
            arg_type = Float64
            nargs = '+'
            default = [0.1]
        "--upper-bounds", "-u"
            help = "upper bounds for initial state"
            arg_type = Float64
            nargs = '+'
            default = [1.]
        "--lower-bounds-params"
            help = "lower bounds for parameters"
            arg_type = Float64
            nargs = '+'
            default = [0.1]
        "--upper-bounds-params"
            help = "upper bounds for parameters"
            arg_type = Float64
            nargs = '+'
            default = [10.]
        "--lower-bounds-obs-params"
            help = "lower bounds for observation parameters"
            arg_type = Float64
            nargs = '+'
            default = [0.1]
        "--upper-bounds-obs-params"
            help = "upper bounds for observation parameters"
            arg_type = Float64
            nargs = '+'
            default = [10.]
        "--system-noise"
            help = "system noise variance"
            arg_type = Float64
            nargs = '+'
            default = [0.0001]
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
            default = [0.01]
        "--obs-iteration"
            help = "observation iteration"
            arg_type = Int
            default = 1
        "--dt"
            help = "Δt"
            arg_type = Float64
            default = 0.125
        "--spinup"
            help = "spinup"
            arg_type = Float64
            default = 0.
        "--duration", "-t"
            help = "assimilation duration"
            arg_type = Float64
            default = 10.
        "--generation-seed", "-s"
            help = "seed for orbit generation"
            arg_type = Int
            default = 0
        "--trials"
            help = "#trials for gradient descent initial value"
            arg_type = Int
            default = 10
        "--regularization-coefficient"
            help = "regularization coefficient"
            arg_type = Float64
            default = 0.
        "--replicates"
            help = "#replicates"
            arg_type = Int
            default = 1
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
            default = collect(0.:0.125:10.)
        "--parameters"
            help = "name of the parameters"
            arg_type = String
            nargs = '+'
            default = ["p"]
        "--obs-parameters"
            help = "name of observation parameters"
            arg_type = String
            nargs = '+'
            default = ["r"]
    end

    parsed_args = parse_args(args, settings; as_symbols=true) # ARGS is needed for static compilation; Otherwise, global ARGS is used.

    # check_args(settings; parsed_args...)

    gradient_covariance_check_obs_m!(dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, observation!, observation_jacobianx!, observation_jacobianr!, observation_hessianxx!, observation_hessianxr!, observation_hessianrr!; parsed_args...)

    return 0
end

julia_main(ARGS)
