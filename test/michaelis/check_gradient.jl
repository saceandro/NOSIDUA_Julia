include("../../src/Adjoints.jl")

module Michaelis

using ArgParse, Adjoints, Distributions, CatViews.CatView

export julia_main

include("../../util/optparser.jl")
include("../../util/check_gradient.jl")
include("model.jl")

@views function gradient_covariance_check_obs_m!(
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
    iter = nothing,
    numerical_differentiation_delta = nothing,
    x0 = nothing
    )

    L = length(initial_lower_bounds)
    N = L - length(true_params)
    model = Model(typeof(obs_variance), N, L, dxdt!, jacobian!, hessian!)
    srand(generation_seed)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L]
    a = Adjoint(dt, duration, obs_variance, x0, copy(true_params), replicates)
    orbit!(a, model)
    tob = deepcopy(a.x)
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
    end

    x0_p = rand.(dists)
    initialize!(a, x0_p)
    gr_ana = orbit_gradient!(a, model)
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

    res_ana, minres = assimilate!(a, model, dists, trials)
    res_num = numerical_covariance!(a, model, numerical_differentiation_delta)
    dir *= "/true_params_$(join(true_params, "_"))/initial_lower_bounds_$(join(initial_lower_bounds, "_"))/initial_upper_bounds_$(join(initial_upper_bounds, "_"))/spinup_$spinup/generation_seed_$generation_seed/trials_$trials/obs_variance_$obs_variance/obs_iteration_$obs_iteration/dt_$dt/duration_$duration/replicates_$replicates/iter_$iter/"
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
            default = [1., 2., 3., 4., 5., 6.]
        "--initial-lower-bounds", "-l"
            help = "lower bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = [0., 0., 0., 1., 2., 3., 4., 5.]
        "--initial-upper-bounds", "-u"
            help = "upper bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = [2., 2., 2., 3., 4., 5., 6., 7.]
        "--obs-variance"
            help = "observation variance"
            arg_type = Float64
            default = 0.0001
        "--obs-iteration"
            help = "observation iteration"
            arg_type = Int
            default = 5
        "--dt"
            help = "Δt"
            arg_type = Float64
            default = 0.01
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
            default = 5
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
            default = [3./201., 1./224.]
    end

    parsed_args = parse_args(args, settings; as_symbols=true) # ARGS is needed for static compilation; Otherwise, global ARGS is used.
    check_args(settings; parsed_args...)

    gradient_covariance_check_obs_m!(dxdt!, jacobian!, hessian!; parsed_args...)

    return 0
end

end
