function set_args(settings)
    for (arg,val) in parse_args(settings)
        @eval (($(Symbol(replace(arg, "-", "_")))) = ($val))
    end
end

function args2varname(parsed_args)
    for (arg,val) in parsed_args
        @eval (($(Symbol(replace(arg, "-", "_")))) = ($val))
    end
end
