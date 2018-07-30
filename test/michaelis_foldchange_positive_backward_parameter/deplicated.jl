# next_x!(        a::Adjoint,       m::Model,       t, x, x_nxt)                                               = (m.dxdt!(    m, t, x, a.p);                                     x_nxt .=  x .+ m.dxdt                                                             .* a.dt; nothing)

# next_dx!(       a::Adjoint,       m::Model,       t, x,        dx, dx_nxt)                                   = (m.jacobian!(m, t, x, a.p);                                    dx_nxt .= dx .+ m.jacobian  * CatView(dx, a.dp)                                    .* a.dt; nothing)

# @views prev_λ!( a::Adjoint{N},    m::Model{N},    t, x,                   λ, λ_prev)            where {N}    = (m.jacobian!(m, t, x, a.p);                                    λ_prev .=  λ .+ m.jacobian' *  λ[1:N]                                              .* a.dt; nothing)

# @views prev_dλ!(a::Adjoint{N, L}, m::Model{N, L}, t, x,        dx,        λ,       dλ, dλ_prev) where {N, L} = (m.hessian!( m, t, x, a.p); prev_λ!(a, m, t, x, dλ, dλ_prev); dλ_prev .+= reshape(reshape(m.hessian, N*L, L) * CatView(dx, a.dp), N, L)' * λ[1:N] .* a.dt; nothing)


# @views function orbit!(a::Adjoint{N,L,K,T}, m) where {N,L,K,T<:AbstractFloat}
#     for _i in 1:a.steps
#         next_x!(a, m, a.dt*(_i-1), a.x[:,_i], a.x[:,_i+1])
#     end
#     obs_variance!(a, m)
#     nothing
# end

# @views function orbit!(a::Adjoint{N,L,K,T}, m) where {N,L,K,T<:AbstractFloat}
#     res = zeros(N)
#     for _i in 1:a.steps
#         next_x!(a, m, a.dt*(_i-1), a.x[:,_i], a.x[:,_i+1], res)
#     end
#     obs_variance!(a, m)
#     nothing
# end

# @views function orbit!(a::Adjoint{N,L,K,T}, m) where {N,L,K,T<:AbstractFloat}
#     # plt = path3d(1, xlim=(-10,10), ylim=(-10,10), zlim=(-10,10), xlab="x1", ylab="x2", zlab="x3", title="Lorenz96", marker=1)
#     res = zeros(N)
#     for _i in 1:a.steps
#         # println(a.x[:,_i])
#         Juno.@enter next_x!(a, m, a.dt*(_i-1), a.x[:,_i], a.x[:,_i+1], res)
#         # push!(plt, a.x[1,_i+1], a.x[2,_i+1], a.x[3,_i+1])
#     end
#     # println(a.x[:,end])
#     # display(plt)
#     obs_variance!(a, m)
#     nothing
# end

# function orbit!(a::Adjoint{N,L,K,T}, m) where {N,L,K,T<:AbstractFloat}
#     model(dx, x, p, t) = (m.dxdt!(m, t, x, p); dx .= m.dxdt)
#     model(::Type{Val{:jac}}, ddx, x, p, t) = (m.jacobianx!(m, t, x, p); ddx .= m.jacobianx)
#     prob = ODEProblem(model, a.x[:,1], (a.t[1], a.t[end]), a.p)
#     # sol = solve(prob, ImplicitEuler(); saveat=a.t)
#     # sol = solve(prob, ImplicitEuler(); dense=false, saveat=a.dt)
#     sol = solve(prob, ImplicitEuler(); adaptive=false, tstops=a.t)
#     println(length(sol.t))
#     # println(size(collect(Iterators.flatten(sol.u))))
#     a.x .= reshape(collect(Iterators.flatten(sol.u)), N, a.steps+1)
#     # for i in 1:a.steps+1
#     #     a.x[:,i] = sol.u[i]
#     # end
#     obs_variance!(a, m)
#     nothing
# end


@views function f!(F, a::Adjoint, m, t, x, x_prev)
    m.dxdt!(m, t, x, a.p)
    F .= x .- m.dxdt .* a.dt .- x_prev
    nothing
end

@views function j!(J, a::Adjoint{N}, m, t, x) where {N}
    m.jacobianx!(m, t, x, a.p)
    J .= eye(N) .- m.jacobianx .* a.dt
    nothing
end


# @views function next_x!(a::Adjoint{N,L,K,T},       m::Model,       t, x_prev, x) where {N,L,K,T}
#     bufx = Array{T}(N)
#     bufF = Array{T}(N)
#     df = OnceDifferentiable((F, x) -> f!(F, a, m, t, x, x_prev), (J, x) -> j!(J, a, m, t, x), bufx, bufF)
#     res = nlsolve(df, x_prev; method=:trust_region, xtol=0., ftol=1e2)
#     if !(res.x_converged || res.f_converged)
#         println(res)
#     end
#     # if (res.x_converged || res.f_converged)
#         x .= res.zero
#     # else
#         # error("next_x calculation failed")
#     # end
#     nothing
# end

# @views function next_x!(a::Adjoint{N,L,K,T},       m::Model,       t, x_prev, x) where {N,L,K,T}
#     bufx = Array{T}(N)
#     bufF = Array{T}(N)
#     # df = OnceDifferentiable((F, x) -> f!(F, a, m, t, x, x_prev), (J, x) -> j!(J, a, m, t, x), bufx, bufF)
#     # minres = nlsolve(df, x_prev; method=:trust_region, xtol=0., ftol=1e-8)
#     minres = nlsolve((F, x) -> f!(F, a, m, t, x, x_prev), copy(x_prev))
#     residual_norm = minres.residual_norm
#
#     trylimit = a.trylimit
#     res = nothing
#     while (!minres.f_converged) && (trylimit > 0)
#         # res = nlsolve(df, x_prev .+ x_prev .* rand.(a.search_box); method=:trust_region, xtol=0., ftol=1e-8)
#         res = nlsolve((F, x) -> f!(F, a, m, t, x, x_prev), copy(x_prev .+ x_prev .* rand.(a.search_box)))
#         if res.residual_norm < minres.residual_norm
#             minres = res
#         end
#         trylimit -= 1
#     end
#     # if !minres.f_converged
#     #     println(minres.residual_norm)
#     # end
#
#     # if (res.x_converged || res.f_converged)
#         x .= minres.zero
#     # else
#         # error("next_x calculation failed")
#     # end
#     nothing
# end

# function next_x!(a::Adjoint{N,L,K,T},       m::Model,       t, x_prev, x) where {N,L,K,T}
#     # bufx = Array{T}(N)
#     # bufF = Array{T}(N)
#     # df = OnceDifferentiable((F, x) -> f!(F, a, m, t, x, x_prev), (J, x) -> j!(J, a, m, t, x), bufx, bufF)
#     # minres = nlsolve(df, x_prev; method=:trust_region, xtol=0., ftol=1e-8)
#     Gadfly.plot(x1 -> f1(a, m, t, x1, x_prev[1]), -5, 5)
#     minres1 = nlsolve((F, x1) -> f1!(F, a, m, t, x1, x_prev[1]), x_prev[1:1])
#     residual_norm1 = minres1.residual_norm
#     Gadfly.plot(x2 -> f2(a, m, t, minres1.zero[1], x_prev[1], x2, x_prev[2]), -5, 5)
#     minres2 = nlsolve((F, x2) -> f2!(F, a, m, t, minres1.zero[1], x_prev[1], x2, x_prev[2]), x_prev[2:2])
#     residual_norm2 = minres2.residual_norm
#     residual_norm = sqrt((residual_norm1^2 + residual_norm2^2)/2.)
#
#     # trylimit = a.trylimit
#     # res = nothing
#     # while (!minres.f_converged) && (trylimit > 0)
#     #     # res = nlsolve(df, x_prev .+ x_prev .* rand.(a.search_box); method=:trust_region, xtol=0., ftol=1e-8)
#     #     res = nlsolve((F, x) -> f!(F, a, m, t, x, x_prev), copy(x_prev .+ x_prev .* rand.(a.search_box)))
#     #     if res.residual_norm < minres.residual_norm
#     #         minres = res
#     #     end
#     #     trylimit -= 1
#     # end
#     if !minres1.f_converged || !minres2.f_converged
#         println(residual_norm)
#     end
#
#     # if (res.x_converged || res.f_converged)
#         x[1] = minres1.zero[1]
#         x[2] = minres2.zero[1]
#     # else
#         # error("next_x calculation failed")
#     # end
#     nothing
# end

@views function next_x!(a::Adjoint{N,L,K,T},       m::Model,       t, x_prev, x, res, x_min, res_min; tol=1e-8, iterlimit=1000) where {N,L,K,T}
    # _res = (x) -> (m.dxdt!(m, t, x, a.p); x .- x_prev .- m.dxdt .* a.dt)
    _res(x) = (m.dxdt!(m, t, x, a.p); x .- x_prev .- m.dxdt .* a.dt)
    _norm_res(x) = norm(_res(x), 2)

    m.dxdt!(m, t, x_prev, a.p) # forward Euler step as first guess
    x .= x_prev .+ m.dxdt .* a.dt
    # println(STDERR, "forward_euler_norm: ", norm_res(x))

    # white_panel = Theme(panel_fill="white")
    # pl = Gadfly.plot(
    # layer(x=x[1], y=x[2], Geom.point),
    # layer(z=(x,y) -> norm_res([x,y]), x=linspace(1e-5, 2.*x_prev[1], 100), y=linspace(1e-5, 2.*x_prev[2], 100), Geom.contour(levels=100)),
    # layer(z=(x,y) -> norm_res([x,y]), x=linspace(2.*x_prev[1], -1e-5, 100), y=linspace(2.*x_prev[2], -1e-5, 100), Geom.contour(levels=100)),
    # white_panel)

    # x_foward_euler = copy(x)

    res .= _res(x)

    norm_res_min = norm(res, 2)
    res_min .= res
    x_min .= x

    if norm(res, 2)<tol
        # println("forward euler is accepted: ", x)
        # println("t = $t,\tx = $x,\tres = $res\tfoward euler is accepted.")
        return nothing
    end

    for i in 1:iterlimit
        m.jacobianx!(m, t, x, a.p)
        x .-= (I - m.jacobianx .* a.dt) \ res
        res .= _res(x)
        # print("iter $i: $x\t$res")
        norm_res = norm(res, 2)
        if norm_res < norm_res_min
            norm_res_min = norm_res
            res_min .= res
            x_min .= x
        end
        if norm(res, 2)<tol
            # println("newton trial 0 iter $i: ", x)
            # pl = Gadfly.plot(
            # layer(z=(x1,x2) -> norm_res([x1,x2,x[3],x[4],x[5]]), x=linspace(x_prev[1]+a.search_box[1].a, x_prev[1]+a.search_box[1].b, 100), y=linspace(x_prev[2]+a.search_box[2].a, x_prev[2]+a.search_box[2].b, 100), Geom.contour(levels=100)),
            # layer(x=[x_foward_euler[1]], y=[x_foward_euler[2]], Geom.point, Theme(default_color=color("red"))),
            # layer(x=[x[1]], y=[x[2]], Geom.point, Theme(default_color=color("blue"))),
            # white_panel)
            # draw(PDF("contourplots/next_x_t$t.pdf", 24cm, 24cm), pl)
            # println("\tconverged")
            # println("t = $t,\tx = $x,\tres = $res\tconverged.")
            return nothing
        end
        # println("")
    end

    # for i=1:a.trylimit
    #     println(x)
    #     res .= _res(x)
    #     m.jacobianx!(m, t, x, a.p)
    #     x .-= (I - m.jacobianx .* a.dt) \ res
    #     if norm(res, 2)<tol
    #         println(x)
    #         # pl = Gadfly.plot(
    #         # layer(z=(x1,x2) -> norm_res([x1,x2,x[3],x[4],x[5]]), x=linspace(x_prev[1]+a.search_box[1].a, x_prev[1]+a.search_box[1].b, 100), y=linspace(x_prev[2]+a.search_box[2].a, x_prev[2]+a.search_box[2].b, 100), Geom.contour(levels=100)),
    #         # layer(x=[x_foward_euler[1]], y=[x_foward_euler[2]], Geom.point, Theme(default_color=color("red"))),
    #         # layer(x=[x[1]], y=[x[2]], Geom.point, Theme(default_color=color("blue"))),
    #         # white_panel)
    #         # draw(PDF("contourplots/next_x_t$t.pdf", 24cm, 24cm), pl)
    #         return nothing
    #     end
    # end

    for j in 1:a.trylimit
        m.dxdt!(m, t, x_prev, a.p) # forward Euler step as first guess
        x .= x_prev .+ m.dxdt .* a.dt .+ rand.(a.search_box)
        res .= _res(x)
        for i in 1:iterlimit
            m.jacobianx!(m, t, x, a.p)
            x .-= (I - m.jacobianx .* a.dt) \ res
            res .= _res(x)
            norm_res = norm(res, 2)
            if norm_res < norm_res_min
                norm_res_min = norm_res
                res_min .= res
                x_min .= x
            end
            if norm(res, 2)<tol
                # println("newton trial $j iter $i: ", x)
                # pl = Gadfly.plot(
                # layer(z=(x1,x2) -> norm_res([x1,x2,x[3],x[4],x[5]]), x=linspace(x_prev[1]+a.search_box[1].a, x_prev[1]+a.search_box[1].b, 100), y=linspace(x_prev[2]+a.search_box[2].a, x_prev[2]+a.search_box[2].b, 100), Geom.contour(levels=100)),
                # layer(x=[x_foward_euler[1]], y=[x_foward_euler[2]], Geom.point, Theme(default_color=color("red"))),
                # layer(x=[x[1]], y=[x[2]], Geom.point, Theme(default_color=color("blue"))),
                # white_panel)
                # draw(PDF("contourplots/next_x_t$t.pdf", 24cm, 24cm), pl)
                # println("t = $t,\tx = $x,\tres = $res\tconverged.")
                return nothing
            end
        end
    end
    error("Newton's method did not converged!")
    # x .= x_min
    # res .= res_min
    # norm_res = norm_res_min
    # pl = Gadfly.plot(
    # layer(z=(x,y) -> _norm_res([x,y]), x=linspace(x_prev[1]+a.search_box[1].a, x_prev[1]+a.search_box[1].b, 100), y=linspace(x_prev[2]+a.search_box[2].a, x_prev[2]+a.search_box[2].b, 100), Geom.contour(levels=100)),
    # layer(x=[x_foward_euler[1]], y=[x_foward_euler[2]], Geom.point, Theme(default_color=color("red"))),
    # layer(x=[x[1]], y=[x[2]], Geom.point, Theme(default_color=color("blue"))),
    # white_panel)
    # draw(PDF("unconverged_contourplots/next_x_t$t.pdf", 24cm, 24cm), pl)
    # println("t = $t,\tx = $x,\tres = $res\tNewton's method did not converged.\tp = $(a.p)")
end
