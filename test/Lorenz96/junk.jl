#!/usr/bin/env julia

using ArgParse

include("../../util/argprase.jl")

function main()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--opt1"
            help = "an option with an argument"
        "--opt2", "-o"
            help = "another option with an argument"
            arg_type = Int
            default = 0
        "--opt3"
            help = "option 3 with an argument"
            nargs = '*'
            arg_type = Float64
            default = [8.,9.,10.]
        "--flag1"
            help = "an option without argument, i.e. a flag"
            action = :store_true
        "arg1"
            help = "a positional argument"
            required = true
    end

    set_args(s)

    println("opt1: ", opt1)
    println("opt2: ", opt2)
    println("opt3: ", opt3)
    println("flag1: ", flag1)
    println("arg1: ", arg1)
    # println("Parsed args:")
    # for (arg,val) in parsed_args
    #     println("  $arg  =>  $val\t(type: $(typeof(val)))")
    # end
    # res = parse.(Float64, split(parsed_args["opt1"], ","))
    # println("opt1:")
    # println("$res\t(type: $(typeof(res)))")
end

main()
