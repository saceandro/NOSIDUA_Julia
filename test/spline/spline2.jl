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

    c = similar(time_point, T-1)
    d = similar(time_point, T-1)

    for _i in 1:T-1
        c[_i] = (-b[_i+1] - 2.*b[_i] + 3.*(control[_i+1] - control[_i])/dt[_i]) / dt[_i]
        d[_i] = ( b[_i+1] +    b[_i] - 2.*(control[_i+1] - control[_i])/dt[_i]) / dt[_i]^2
    end

    println("b:\t", b)
    println("c:\t", c)
    println("d:\t", d)


    println("connectivity check:")
    f = similar(time_point, T-1)
    for _i in 1:T-1
        f[_i] = control[_i] + b[_i]*dt[_i] + c[_i]*dt[_i]^2 + d[_i]*dt[_i]^3
    end
    println(control[2:end])
    println(f)


    println("gradneit connectivity check:")
    e = similar(time_point, T-1)
    for _i in 1:T-1
        e[_i] = b[_i] + 2.*c[_i]*dt[_i] + 3.*d[_i]*dt[_i]^2
    end
    println(b[2:end])
    println(e)


    println("initial curvature:\t", c[1])
    println("end curvature:\t", 2.*c[end] + 6.*d[end]*dt[end])

    println("curvature connectivity check:")
    a = similar(time_point, T-2)
    for _i in 1:T-2
        a[_i] = 2.*c[_i] + 6.*d[_i]*dt[_i]
    end
    println(a)
    println(2.*c[2:T-1])


    function spline_curve(t)
        for _i in 1:T-1
            if t <= time_point[_i+1]
                delta = t - time_point[_i]
                return control[_i] + b[_i]*delta + c[_i]*delta^2 + d[_i]*delta^3
            end
        end
    end

    function gradient_curve(t)
        for _i in 1:T-1
            if t <= time_point[_i+1]
                delta = t - time_point[_i]
                return b[_i] + 2.*c[_i]*delta + 3.*d[_i]*delta^2
            end
        end
    end

    function curvature_curve(t)
        for _i in 1:T-1
            if t <= time_point[_i+1]
                delta = t - time_point[_i]
                return 2.*c[_i] + 6.*d[_i]*delta
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
    end

    parsed_args = parse_args(args, settings; as_symbols=true) # ARGS is needed for static compilation; Otherwise, global ARGS is used.
    calc_spline!(; parsed_args...)
end

julia_main_test(ARGS)
