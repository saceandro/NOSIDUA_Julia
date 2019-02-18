@views function u(m, t, q)
    T = length(m.time_point)
    for _i in 1:T-1
        if t <= m.time_point[_i+1]
            Δ = t - m.time_point[_i]
            φ = 3Δ^2/m.Δtime_point[_i]^2 - 2Δ^3/m.Δtime_point[_i]^3
            ν = Δ - 2Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
            τ = -Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
            return (1 - φ) * q[_i] + φ * q[_i+1] + ν * m.eqdot[_i] + τ * m.eqdot[_i+1]
        end
    end
end

@views function dxdt!(m::Model, t, x, p)
    I = u(m, t, p[2:end])
    m.dxdt[1] = -p[1] * x[1] + I
    # m.dxdt[1] = -p[1] * x[1] + p[3]
    nothing
end

@views function jacobianx!(m::Model, t, x, p) # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    m.jacobianx[1,1] = -p[1]
    nothing
end

@views function jacobianp!(m::Model, t, x, p)
    fill!(m.jacobianp, 0.) # bug fixed

    m.jacobianp[1,1] = -x[1]

    if t == m.time_point[1]
        m.jacobianp[1,2] = 1.
        return nothing
    end

    T = length(m.time_point)
    for _i in 1:T-1
        if m.time_point[_i] < t && t < m.time_point[_i+1]
            Δ = t - m.time_point[_i]
            φ = 3Δ^2/m.Δtime_point[_i]^2 - 2Δ^3/m.Δtime_point[_i]^3
            ν = Δ - 2Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
            τ = -Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2

            for _j in 1:T
                m.jacobianp[1,_j+1] = ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]
            end
            m.jacobianp[1,_i+1]  += 1 - φ
            m.jacobianp[1,_i+2]  +=     φ
            break
        end

        if t == m.time_point[_i+1]
            m.jacobianp[1,_i+2] = 1.
            break
        end
    end

    # if (t == m.time_point[2])
    #     m.jacobianp[1,3] = 1.
    # end

    nothing
end

@views function hessianxx!(m::Model, t, x, p)
    nothing
end

function hessianxp!(m::Model, t, x, p)
    m.hessianxp[1,1,1] = -1.
    nothing
end

function hessianpp!(m::Model, t, x, p)
    nothing
end

observation(x) = x

d_observation(x) = 1.

dd_observation(x) = 0.

inv_observation(z) = z

function calc_eqdot!(m::Model, q)
    T = length(m.time_point)
    m.rhs[1] = (q[2] - q[1]) / m.Δtime_point[1]
    for _i in 2:T-1
        m.rhs[_i] = (q[_i] - q[_i-1]) / m.Δtime_point[_i-1]^2 + (q[_i+1] - q[_i]) / m.Δtime_point[_i]^2
    end
    m.rhs[T] = (q[T] - q[T-1]) / m.Δtime_point[T-1]
    m.rhs .*= 3.

    m.eqdot .= m.spline_lhs \ m.rhs
end
