@views function u(m, t, q)
    T = length(m.time_point)
    for _i in 1:T-1
        if t <= m.time_point[_i+1]
            Δ = t - m.time_point[_i]
            φ = 3Δ^2/m.Δtime_point[_i]^2 - 2Δ^3/m.Δtime_point[_i]^3
            ν = Δ - 2Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
            τ = -Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
            return (1 - φ) * exp(q[_i]) + φ * exp(q[_i+1]) + ν * m.eqdot[_i] + τ * m.eqdot[_i+1]
        end
    end
end

@views function dxdt!(m::Model, t, x, p)
    I = u(m, t, p[2:4])
    ex = exp.(x)
    ep = exp.(p)
    m.dxdt[1] = -ep[1] + I/ex[1]
    nothing
end

@views function jacobianx!(m::Model, t, x, p) # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    I = u(m, t, p[2:4])
    ex = exp.(x)
    m.jacobianx[1,1] = -I/ex[1]
    nothing
end

@views function jacobianp!(m::Model, t, x, p)
    fill!(m.jacobianp, 0.) # bug fixed
    I = u(m, t, p[2:4])
    ex = exp.(x)
    ep = exp.(p)

    m.jacobianp[1,1] = -ep[1]

    # if t == m.time_point[1]
    #     m.jacobianp[1,2] = ep[2]/ex[1]
    #     return nothing
    # end

    T = length(m.time_point)
    for _i in 1:T-1
        if m.time_point[_i] <= t && t <= m.time_point[_i+1]
            Δ = t - m.time_point[_i]
            φ = 3Δ^2/m.Δtime_point[_i]^2 - 2Δ^3/m.Δtime_point[_i]^3
            ν = Δ - 2Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
            τ = -Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2

            for _j in 1:T
                m.jacobianp[1,_j+1] = (ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]) * ep[_j+1]/ex[1]
            end
            m.jacobianp[1,_i+1]  += (1 - φ) * ep[_i+1]/ex[1]
            m.jacobianp[1,_i+2]  +=      φ  * ep[_i+2]/ex[1]
            break
        end

        # if t == m.time_point[_i+1]
        #     m.jacobianp[1,_i+2] = ep[_i+2]/ex[1]
        #     break
        # end
    end
    nothing
end

@views function hessianxx!(m::Model, t, x, p)
    I = u(m, t, p[2:4])
    ex = exp.(x)
    m.hessianxx[1,1,1] = I/ex[1]
    nothing
end

function hessianxp!(m::Model, t, x, p)
    fill!(m.hessianxp, 0.) # bug fixed
    ex = exp.(x)
    ep = exp.(p)

    # if t == m.time_point[1]
    #     m.hessianxp[1,2,2] = ep[2]/ex[1]
    #     return nothing
    # end

    T = length(m.time_point)
    for _i in 1:T-1
        if m.time_point[_i] <= t && t <= m.time_point[_i+1]
            Δ = t - m.time_point[_i]
            φ = 3Δ^2/m.Δtime_point[_i]^2 - 2Δ^3/m.Δtime_point[_i]^3
            ν = Δ - 2Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
            τ = -Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2

            for _j in 1:T
                m.hessianxp[1,1,_j+1] = -(ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]) * ep[_j+1]/ex[1]
            end
            m.hessianxp[1,1,_i+1] -= (1 - φ) * ep[_i+1]/ex[1]
            m.hessianxp[1,1,_i+2] -=      φ  * ep[_i+2]/ex[1]
            break
        end

        # if t == m.time_point[_i+1]
        #     m.hessianxp[1,_i+2,_i+2] = ep[_i+2]/ex[1]
        #     break
        # end
    end
    nothing
end

function hessianpp!(m::Model, t, x, p)
    fill!(m.hessianpp, 0.) # bug fixed
    ex = exp.(x)
    ep = exp.(p)

    m.hessianpp[1,1,1] = -ep[1]

    # if t == m.time_point[1]
    #     m.hessianpp[1,2,2] = ep[2]/ex[1]
    #     return nothing
    # end

    T = length(m.time_point)
    for _i in 1:T-1
        if m.time_point[_i] <= t && t <= m.time_point[_i+1]
            Δ = t - m.time_point[_i]
            φ = 3Δ^2/m.Δtime_point[_i]^2 - 2Δ^3/m.Δtime_point[_i]^3
            ν = Δ - 2Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
            τ = -Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2

            for _j in 1:T
                m.hessianpp[1,_j+1,_j+1] = (ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]) * ep[_j+1]/ex[1]
            end
            m.hessianpp[1,_i+1,_i+1] += (1 - φ) * ep[_i+1]/ex[1]
            m.hessianpp[1,_i+2,_i+2] +=      φ  * ep[_i+2]/ex[1]
            break
        end

        # if t == m.time_point[_i+1]
        #     m.hessianpp[1,_i+2,_i+2] = ep[_i+2]/ex[1]
        #     break
        # end
    end
    nothing
end

function observation!(m::Model, t, x, p)
    ex = exp.(x)
    m.observation[1] = p[5] + p[6] * ex[1]
    m.observation[2] = u(m, t, p[2:4])
end

function observation_jacobianx!(m::Model, t, x, p)
    ex = exp.(x)
    m.observation_jacobianx[1,1] = p[6] * ex[1]
end

function observation_jacobianp!(m::Model, t, x, p)
    fill!(m.observation_jacobianp, 0.)
    ep = exp.(p)

    T = length(m.time_point)
    for _i in 1:T-1
        if m.time_point[_i] <= t && t <= m.time_point[_i+1]
            Δ = t - m.time_point[_i]
            φ = 3Δ^2/m.Δtime_point[_i]^2 - 2Δ^3/m.Δtime_point[_i]^3
            ν = Δ - 2Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
            τ = -Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2

            for _j in 1:T
                m.observation_jacobianp[2,_j+1] = (ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]) * ep[_j+1]
            end
            m.observation_jacobianp[2,_i+1] += (1 - φ) * ep[_i+1]
            m.observation_jacobianp[2,_i+2] +=      φ  * ep[_i+2]
            break
        end
    # for _i in 1:T
    #     if t == m.time_point[_i]
    #         m.observation_jacobianp[2,_i+1] = ep[_i+1]
    #         break
    #     end
    end
end

function observation_jacobianr!(m::Model, t, x, p)
    ex = exp.(x)
    m.observation_jacobianr[1,1] = 1.
    m.observation_jacobianr[1,2] = ex[1]
end

function observation_hessianxx!(m::Model, t, x, p)
    ex = exp.(x)
    m.observation_hessianxx[1,1,1] = p[6] * ex[1]
end

function observation_hessianxr!(m::Model, t, x, p)
    ex = exp.(x)
    m.observation_hessianxr[1,1,2] = ex[1]
end

function observation_hessianpp!(m::Model, t, x, p)
    fill!(m.observation_hessianpp, 0.)
    ep = exp.(p)

    T = length(m.time_point)
    for _i in 1:T-1
        if m.time_point[_i] <= t && t <= m.time_point[_i+1]
            Δ = t - m.time_point[_i]
            φ = 3Δ^2/m.Δtime_point[_i]^2 - 2Δ^3/m.Δtime_point[_i]^3
            ν = Δ - 2Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
            τ = -Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2

            for _j in 1:T
                m.observation_hessianpp[2,_j+1,_j+1] = (ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]) * ep[_j+1]
            end
            m.observation_hessianpp[2,_i+1,_i+1] += (1 - φ) * ep[_i+1]
            m.observation_hessianpp[2,_i+2,_i+2] +=      φ  * ep[_i+2]
            break
        end
    end
end

function observation_hessianrr!(m::Model, t, x, p)
end

# observation(x, r) = r[1] + r[2]*exp(x[1])
#
# d_observation(x, r) = r[2]*exp(x[1])
#
# dd_observation(x, r) = r[2]*exp(x[1])
#
# inv_observation(z, r) = log((z[1]-r[1])/r[2])
#
# function dr_observation!(m::Model, x, r)
#     ex = exp.(x)
#
#     m.dr_observation[1,1] = 1.
#     m.dr_observation[1,2] = ex[1]
#     nothing
# end

function calc_eqdot!(m::Model, q)
    eq = exp.(q)
    T = length(m.time_point)
    m.rhs[1] = (eq[2] - eq[1]) / m.Δtime_point[1]
    for _i in 2:T-1
        m.rhs[_i] = (eq[_i] - eq[_i-1]) / m.Δtime_point[_i-1]^2 + (eq[_i+1] - eq[_i]) / m.Δtime_point[_i]^2
    end
    m.rhs[T] = (eq[T] - eq[T-1]) / m.Δtime_point[T-1]
    m.rhs .*= 3.

    m.eqdot .= m.spline_lhs \ m.rhs
end
