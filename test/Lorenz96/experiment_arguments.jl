include("../../src/Adjoints.jl")

module Experiment

using Adjoints, Distributions, ArgParse, CatViews.CatView, Juno

export julia_main

include("model.jl")
include("../../util/argprase.jl")
include("../../util/experiment_ccompile.jl")

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    settings = ArgParseSettings("Adjoint method",
                                prog = "experiment_arguments",
                                version = "$(basename(@__FILE__)) version 0.1",
                                add_version = true)

    @add_arg_table settings begin
        "--dir", "-d"
            help = "output directory"
            default = "result1/"
        "--dimension", "-n"
            help = "model dimension"
            arg_type = Int
            default = 5
        "--true-params", "-p"
            help = "true parameters"
            arg_type = Float64
            nargs = '*'
            default = [8.,1.]
        "--initial-lower-bounds", "-l"
            help = "lower bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = [-10.,-10.,-10.,-10.,-10.,0.,0.]
        "--initial-upper-bounds", "-u"
            help = "upper bounds for initial state and parameters"
            arg_type = Float64
            nargs = '+'
            default = [10.,10.,10.,10.,10.,16.,2.]
        "--obs-variance"
            help = "observation variance"
            arg_type = Float64
            default = 1.
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
            default = 73.
        "--duration", "-t"
            help = "assimilation duration"
            arg_type = Float64
            default = 1.
        "--generation-seed", "-s"
            help = "seed for orbit generation"
            arg_type = Int
            default = 0
        "--trials"
            help = "#trials for gradient descent initial value"
            arg_type = Int
            default = 50
        "--replicates"
            help = "#replicates"
            arg_type = Int
            default = 1
        "--iter"
            help = "#iterations"
            arg_type = Int
            default = 2
    end

    # set_args(settings)
    parsed_args = parse_args(ARGS, settings) # ARGS is needed for static compilation; Otherwise, global ARGS is used.

    # args_hash = hash([dimension, true_params, initial_lower_bounds, initial_upper_bounds, obs_variance, obs_iteration, dt, spinup, duration, generation_seed, trials, replicates, iter])

    # model = Model(Float64, dimension, length(initial_lower_bounds), dxdt!, jacobian!, hessian!)
    model = Model(Float64, parsed_args["dimension"], length(parsed_args["initial-lower-bounds"]), dxdt!, jacobian!, hessian!)

    # twin_experiment!(args_hash, dir, model, obs_variance, obs_iteration, dt, spinup, duration, generation_seed, true_params, initial_lower_bounds, initial_upper_bounds, replicates, iter, trials)
    twin_experiment!(model, parsed_args)
    return 0
end

end
