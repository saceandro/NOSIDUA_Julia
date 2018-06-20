include("../../src/AdjointsConstraint.jl")

module Michaelis

using ArgParse, AdjointsConstraint, Distributions, CatViews.CatView, Juno

export julia_main

include("../../util/optparser.jl")
include("../../util/check_gradient.jl")
include("model.jl")

# function plot_orbit(dir, a::Adjoint{N}, tob) where {N}
#     white_panel = Theme(panel_fill="white")
#     p_stack = Array{Gadfly.Plot}(0)
#     for _i in 1:N
#         df_tob = DataFrame(t=a.t, x=tob[_i,:], data_type="true orbit")
#         p_stack = vcat(p_stack,
#         Gadfly.plot(
#         layer(df_tob, x="t", y="x", color=:data_type, Geom.line),
#         Guide.xlabel("<i>t</i>"),
#         Guide.ylabel("<i>x<sub>$_i</sub></i>"),
#         white_panel))
#     end
#     draw(PDF(dir * "true_orbit.pdf", 24cm, 40cm), vstack(p_stack))
#     nothing
#     # set_default_plot_size(24cm, 40cm)
#     # vstack(p_stack)
# end

@views function gradient_covariance_check_obs_m!(
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
    pseudo_obs_TSS = nothing,
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
    x0 = nothing
    )

    obs_variance_bak = copy(obs_variance)

    L = length(initial_lower_bounds)
    N = L - length(true_params)
    model = Model(typeof(dt), N, L, dxdt!, jacobian!, jacobian0!, hessian!, hessian0!, hessian00!)
    srand(generation_seed)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L]
    a = Adjoint(dt, duration, obs_variance, pseudo_obs, pseudo_obs_TSS, x0, copy(true_params), replicates)
    orbit!(a, model)
    tob = deepcopy(a.x)
    dir *= "/true_params_$(join(true_params, "_"))/initial_lower_bounds_$(join(initial_lower_bounds, "_"))/initial_upper_bounds_$(join(initial_upper_bounds, "_"))/pseudo_obs_$(join(pseudo_obs, "_"))/pseudo_obs_TSS_$(join(pseudo_obs_TSS, "_"))/spinup_$spinup/generation_seed_$generation_seed/trials_$trials/obs_variance_$(join(obs_variance_bak, "_"))/obs_iteration_$obs_iteration/dt_$dt/duration_$duration/replicates_$replicates/iter_$iter/"
    # mkpath(dir)
    # plot_orbit(dir, a, tob)

    srand(hash([true_params, initial_lower_bounds, initial_upper_bounds, pseudo_obs, pseudo_obs_TSS, obs_variance_bak, obs_iteration, dt, spinup, duration, generation_seed, trials, replicates, iter]))
    d = Normal.(0., sqrt.(obs_variance_bak))
    # a.obs[1,:,:] .= NaN
    # for _replicate in 1:replicates
    #     a.obs[2,:,_replicate] .= view(a.x, 2, :) .+ rand(d[2], a.steps+1)
    #     for _i in 1:obs_iteration:a.steps
    #         for _k in 1:obs_iteration-1
    #             a.obs[2, _i + _k, _replicate] = NaN
    #         end
    #     end
    # end
    for _j in 1:N
        for _replicate in 1:replicates
            a.obs[_j,:,_replicate] .= view(a.x, _j, :) .+ rand(d[_j], a.steps+1)
            for _i in 1:obs_iteration:a.steps
                for _k in 1:obs_iteration-1
                    a.obs[_j, _i + _k, _replicate] = NaN
                end
            end
        end
    end
    for _j in 1:N
        a.Nobs[_j] .+= count(isfinite.(a.obs[_j,:,:]))
    end
    obs_mean_var!(a, model)

    x0_p = rand.(dists)
    initialize!(a, x0_p)
    gr_ana = Vector{typeof(dt)}(L)
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

    # if !isnull(res_ana.covariance) && !isnull(res_num.covariance)
    #     cov_ana = get(res_ana.covariance)
    #     cov_num = get(res_num.covariance)
    #     println("analytical covariance:")
    #     println(cov_ana)
    #     println("numerical covariance:")
    #     println(cov_num)
    #     diff = cov_ana .- cov_num
    #     println("absolute difference:")
    #     println(diff)
    #     println("max absolute difference: ", maximum(abs, diff))
    #     if !any(res_num.covariance == 0)
    #         rel_diff = diff ./ cov_num
    #         println("relative difference")
    #         println(rel_diff)
    #         println("max_relative_difference:\t", maximum(abs, rel_diff))
    #     end
    # end

    if !isnull(res_ana.stddev) && !isnull(res_num.stddev)
        cov_ana = get(res_ana.stddev)
        cov_num = get(res_num.stddev)
        println("analytical stddev:\t", cov_ana)
        println("numerical stddev:\t", cov_num)
        diff = cov_ana .- cov_num
        println("absolute difference:\t", diff)
        println("max absolute difference:\t", maximum(abs, diff))
        if !any(res_num.stddev == 0)
            rel_diff = diff ./ cov_num
            println("relative difference:\t", rel_diff)
            println("max_relative_difference:\t", maximum(abs, rel_diff))
        end
        plot_twin_experiment_result(dir, a, tob, cov_ana)
    else
        plot_twin_experiment_result_wo_errorbar(dir, a, tob)
    end
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
            default = [3., 4.]
        "--initial-lower-bounds", "-l"
            help = "lower bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = [0., 0., 2., 3.]
        "--initial-upper-bounds", "-u"
            help = "upper bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = [2., 2., 4., 5.]
        "--pseudo-obs"
            help = "#pseudo observations"
            arg_type = Int
            nargs = '+'
            default = [0, 0]
        "--pseudo-obs-TSS"
            help = "TSS of pseudo observations"
            arg_type = Float64
            nargs = '+'
            default = [0., 0.]
        "--obs-variance"
            help = "observation variance"
            arg_type = Float64
            nargs = '+'
            default = [0.1, 0.01]
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
        "--generation-seed", "-s"
            help = "seed for orbit generation"
            arg_type = Int
            default = 0
        "--trials"
            help = "#trials for gradient descent initial value"
            arg_type = Int
            default = 10
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
            default = 0.00001
        "--x0"
            help = "initial x"
            arg_type = Float64
            nargs = '+'
            default = [10./11., 40./43.]
    end

    parsed_args = parse_args(args, settings; as_symbols=true) # ARGS is needed for static compilation; Otherwise, global ARGS is used.
    check_args(settings; parsed_args...)

    gradient_covariance_check_obs_m!(dxdt!, jacobian!, jacobian0!, hessian!, hessian0!, hessian00!; parsed_args...)

    return 0
end

end
