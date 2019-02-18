using Gadfly, ArgParse, Juno

@views function calc_spline!(
    ;
    time_point = nothing,
    control = nothing
    )

    T = length(time_point)
    dt = similar(time_point, T-1)
    for _i in 1:T-1
        dt[_i] = time_point[_i+1] - time_point[_i]
    end

    d = similar(time_point, T)
    dl = similar(time_point, T-1)
    du = similar(time_point, T-1)

    d[1] = 1.
    for _i in 2:T-1
        d[_i] = inv(dt[_i-1]) + inv(dt[_i])
    end
    d[T] = 1.
    d .*= 2.

    for _i in 1:T-2
        dl[_i] = inv(dt[_i])
    end
    dl[T-1] = 1.

    du[1] = 1.
    for _i in 2:T-1
        du[_i] = inv(dt[_i])
    end

    M = Tridiagonal(dl, d, du)
    LU = factorize(M)

    rhs = similar(time_point, T)
    rhs[1] = (control[2] - control[1]) / dt[1]
    for _i in 2:T-1
        rhs[_i] = (control[_i] - control[_i-1]) / dt[_i-1]^2 + (control[_i+1] - control[_i]) / dt[_i]^2
    end
    rhs[T] = (control[T] - control[T-1]) / dt[T-1]
    rhs .*= 3.

    b = LU \ rhs

    function spline_curve(t)
        for _i in 1:T-1
            if t <= time_point[_i+1]
                Δ = t - time_point[_i]
                Δi = time_point[_i+1] - time_point[_i]
                φ = 3Δ^2/Δi^2 - 2Δ^3/Δi^3
                υ = Δ - 2Δ^2/Δi + Δ^3/Δi^2
                τ = -Δ^2/Δi + Δ^3/Δi^2
                return (1 - φ) * control[_i] + φ*control[_i+1] + υ*b[_i] + τ*b[_i+1]
            end
        end
    end

    function gradient_curve(t)
        for _i in 1:T-1
            if t <= time_point[_i+1]
                Δ = t - time_point[_i]
                Δi = time_point[_i+1] - time_point[_i]
                φ = 6(Δ/Δi^2 - Δ^2/Δi^3)
                υ = 1 - 4Δ/Δi + 3Δ^2/Δi^2
                τ = -2Δ/Δi + 3Δ^2/Δi^2
                return φ * (control[_i+1] - control[_i]) + υ*b[_i] + τ*b[_i+1]
            end
        end
    end

    function curvature_curve(t)
        for _i in 1:T-1
            if t <= time_point[_i+1]
                Δ = t - time_point[_i]
                Δi = time_point[_i+1] - time_point[_i]
                φ = 6/Δi^2 - 12Δ/Δi^3
                υ = -4/Δi + 6Δ/Δi^2
                τ = -2/Δi + 6Δ/Δi^2
                return φ * (control[_i+1] - control[_i]) + υ*b[_i] + τ*b[_i+1]
            end
        end
    end

    plot(layer(spline_curve, time_point[1], time_point[end], Geom.line, Theme(default_color=color("orange"))),
         layer(x=time_point, y=control, Geom.point, Theme(default_color=color("orange"))),
         layer(gradient_curve, time_point[1], time_point[end], Geom.line, Theme(default_color=color("blue"))),
         layer(curvature_curve, time_point[1], time_point[end], Geom.line, Theme(default_color=color("green"))))
end

Base.@ccallable function julia_main(args::Vector{String})::Cint
    settings = ArgParseSettings("Spline calculator",
                                prog = first(splitext(basename(@__FILE__))),
                                version = "$(first(splitext(basename(@__FILE__)))) version 0.1",
                                add_version = true,
                                autofix_names = true)

    @add_arg_table settings begin
        "--time-point", "-t"
            help = "time points"
            arg_type = Float64
            nargs = '+'
            default = [0., 5., 10., 20., 60.]
        "--control", "-c"
            help = "control points"
            arg_type = Float64
            nargs = '+'
            default = [0., 0.9, 0.8, 0.5, 0.25]
    end

    parsed_args = parse_args(args, settings; as_symbols=true) # ARGS is needed for static compilation; Otherwise, global ARGS is used.
    calc_spline!(; parsed_args...)

    return 0
end

function julia_main_test(args::Vector{String})
    settings = ArgParseSettings("Spline calculator",
                                prog = first(splitext(basename(@__FILE__))),
                                version = "$(first(splitext(basename(@__FILE__)))) version 0.1",
                                add_version = true,
                                autofix_names = true)

    @add_arg_table settings begin
        "--time-point", "-t"
            help = "time points"
            arg_type = Float64
            nargs = '+'
            default = [0., 5., 10., 20., 60.]
        "--control", "-c"
            help = "control points"
            arg_type = Float64
            nargs = '+'
            default = [0., 0.9, 0.8, 0.5, 0.25]
            # default = [0., 1.0, 0.7, 0.3, 0.1]
    end

    parsed_args = parse_args(args, settings; as_symbols=true) # ARGS is needed for static compilation; Otherwise, global ARGS is used.
    calc_spline!(; parsed_args...)
end

julia_main_test(ARGS)
