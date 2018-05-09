include("../../src/Adjoints.jl")

module Experiment

using Adjoints, Distributions, ArgParse, CatViews.CatView

export julia_main

include("model.jl")
include("../../util/experiment_ccompile.jl")
include("../../util/check_args.jl")

Base.@ccallable function julia_main(args::Vector{String})::Cint
    settings = ArgParseSettings("Adjoint method",
                                prog = first(splitext(basename(@__FILE__))),
                                version = "$(first(splitext(basename(@__FILE__)))) version 0.1",
                                add_version = true,
                                autofix_names = true)

    @add_arg_table settings begin
        "--dir", "-d"
            help = "output directory"
            # default = "result1/"
            default = "result1"
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
            default = 1
    end

    parsed_args = parse_args(args, settings; as_symbols=true) # ARGS is needed for static compilation; Otherwise, global ARGS is used.
    check_args(settings; parsed_args...)

    twin_experiment!(dxdt!, jacobian!, hessian!; parsed_args...)

    return 0
end

end
