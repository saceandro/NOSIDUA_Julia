using DifferentialEquations

function parameterized_lorenz(du,u,p,t)
    du[1] = p[1]*(u[2]-u[1])
    du[2] = u[1]*(p[2]-u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
end

# function parameterized_lorenz(u,p,t)
#     [p[1]*(u[2]-u[1]), u[1]*(p[2]-u[3]) - u[2], u[1]*u[2] - p[3]*u[3]]
# end

function parameterized_lorenz(::Type{Val{:jac}}, ddu, u, p, t)
    ddu[1,1] = -p[1]
    ddu[1,2] = p[1]
    ddu[1,3] = 0.
    ddu[2,1] = p[2] - u[3]
    ddu[2,2] = -1.
    ddu[2,3] = -u[1]
    ddu[3,1] = u[2]
    ddu[3,2] = u[1]
    ddu[3,3] = -p[3]
end

function parameterized_lorenz(::Val{:jac}, ddu, u, p, t)
    ddu[1,1] = -p[1]
    ddu[1,2] = p[1]
    ddu[1,3] = 0.
    ddu[2,1] = p[2] - u[3]
    ddu[2,2] = -1.
    ddu[2,3] = -u[1]
    ddu[3,1] = u[2]
    ddu[3,2] = u[1]
    ddu[3,3] = -p[3]
end

# function parameterized_lorenz(::Val{:jac}, u, p, t)
#     ddu = Matrix{Float64}(3,3)
#     ddu[1,1] = -p[1]
#     ddu[1,2] = p[1]
#     ddu[1,3] = 0.
#     ddu[2,1] = p[2] - u[3]
#     ddu[2,2] = -1.
#     ddu[2,3] = -u[1]
#     ddu[3,1] = u[2]
#     ddu[3,2] = u[1]
#     ddu[3,3] = -p[3]
#     ddu
# end

u0 = [1., 0., 0.]
tspan = (0., 100.)
p = [10.0,28.0,8/3]

prob = ODEProblem(parameterized_lorenz,u0,tspan,p)
# sol = solve(prob, ImplicitEuler(); saveat=0.01)
# println(sol)
# plot(sol, vars=(1,2,3))

# sol_bwdeuler = solve(prob, DiffEqBase.InternalEuler.BwdEulerAlg(); tstops=0:0.01:1)
# println(sol_bwdeuler)

# function solve_buff(prob::AbstractODEProblem{uType,tType,isinplace},
#                           Alg::DiffEqBase.InternalEuler.BwdEulerAlg;
#                           dt=(prob.tspan[2]-prob.tspan[1])/100,
#                           tstops=tType[],
#                           tol=1e-5,
#                           maxiter=100,
#                           kwargs... # ignored kwargs
#                           ) where {uType,tType,isinplace}
#     u0 = prob.u0
#     f = prob.f
#     tspan = prob.tspan
#     p = prob.p
#     # TODO: fix numparameters as it picks up the Jacobian
# #    @assert !isinplace "Only out of place functions supported"
#     @assert DiffEqBase.has_jac(f) "Provide Jacobian as f(::Val{:jac}, ...)"
#     du = similar(u0)
#     ddu = similar(u0, length(u0), length(u0))
#     jac = (ddu,u,p,t) -> f(Val{:jac}(),ddu,u,p,t)
#
#     if isempty(tstops)
#         tstops = tspan[1]:dt:tspan[2]
#     end
#     @assert tstops[1]==tspan[1]
#
#     nt = length(tstops)
#     out = Vector{uType}(nt)
#     out[1] = copy(u0)
#     for i=2:nt
#         t = tstops[i]
#         dt = t-tstops[i-1]
#         out[i] = newton(t, dt, out[i-1], p, f, du, jac, ddu, tol, maxiter)
#     end
#     # make solution type
#     build_solution(prob, Alg, tstops, out)
# end

function solve_buff(prob::AbstractODEProblem{uType,tType,isinplace},
                          Alg::DiffEqBase.InternalEuler.BwdEulerAlg;
                          dt=(prob.tspan[2]-prob.tspan[1])/100,
                          tstops=tType[],
                          tol=1e-5,
                          maxiter=100,
                          kwargs... # ignored kwargs
                          ) where {uType,tType,isinplace}
    u0 = prob.u0
    f = prob.f
    tspan = prob.tspan
    p = prob.p
    # TODO: fix numparameters as it picks up the Jacobian
#    @assert !isinplace "Only out of place functions supported"
    @assert DiffEqBase.has_jac(f) "Provide Jacobian as f(::Val{:jac}, ...)"
    res = similar(u0)
    du = similar(u0)
    ddu = similar(u0, length(u0), length(u0))
    jac = (ddu,u,p,t) -> f(Val{:jac}(),ddu,u,p,t)

    if isempty(tstops)
        tstops = tspan[1]:dt:tspan[2]
    end
    @assert tstops[1]==tspan[1]

    nt = length(tstops)
    out = Vector{uType}(nt)
    out[1] = copy(u0)
    for i=2:nt
        t = tstops[i]
        dt = t-tstops[i-1]
        out[i] = newton(t, dt, out[i-1], p, f, du, jac, ddu, res, tol, maxiter)
    end
    # make solution type
    build_solution(prob, Alg, tstops, out)
end

# function newton(t, dt, u_last, p, f, du, jac, ddu, tol, maxiter)
#     res = (u) -> (f(du, u, p, t); u .- u_last .- dt*du)
#     f(du, u_last, p, t) # forward Euler step as first guess
#     u = u_last .+ dt*du
#     for i=1:maxiter
#         jac(ddu, u, p, t)
#         du = -(I - dt*ddu)\res(u)
#         u += du
#         norm(du, Inf)<tol && return u
#     end
#     error("Newton not converged")
# end

function newton(t, dt, u_last, p, f, du, jac, ddu, res, tol, maxiter)
    _res = (u) -> (f(du, u, p, t); u .- u_last .- dt*du)
    f(du, u_last, p, t) # forward Euler step as first guess
    u = u_last .+ dt*du
    for i=1:maxiter
        res .= _res(u)
        jac(ddu, u, p, t)
        du = -(I - dt*ddu)\res
        u += du
        norm(res, 2)<tol && return u
    end
    error("Newton not converged")
end

sol_bwdeuler = solve_buff(prob, DiffEqBase.InternalEuler.BwdEulerAlg(); tstops=0:0.01:1)
println(sol_bwdeuler)
