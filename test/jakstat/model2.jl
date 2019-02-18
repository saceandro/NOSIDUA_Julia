ρ = 0.45/1.4

@views function u2(m, t, q)
    T = length(m.time_point)
    for _i in 1:T-1
        if t <= m.time_point[_i+1]
            Δ = t - m.time_point[_i]
            c = -m.qdot[_i+1] - 2m.qdot[_i] + 3(q[_i+1] - q[_i])/m.Δtime_point[_i]
            d =  m.qdot[_i+1] +  m.qdot[_i] - 2(q[_i+1] - q[_i])/m.Δtime_point[_i]

            return q[_i] + Δ * m.qdot[_i] + Δ^2/m.Δtime_point[_i] * c + Δ^3/m.Δtime_point[_i]^2 * d
        end
    end
end

@views function u(m, t, q)
    T = length(m.time_point)
    for _i in 1:T-1
        if t <= m.time_point[_i+1]
            Δ = t - m.time_point[_i]
            φ = 3Δ^2/m.Δtime_point[_i]^2 - 2Δ^3/m.Δtime_point[_i]^3
            ν = Δ - 2Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
            τ = -Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
            return (1 - φ) * q[_i] + φ * q[_i+1] + ν * m.qdot[_i] + τ * m.qdot[_i+1]
        end
    end
end

@views function du2(m, t, q)
    T = length(m.time_point)
    for _i in 1:T-1
        if t <= m.time_point[_i+1]
            Δ = t - m.time_point[_i]
            φ = 6 * (Δ/m.Δtime_point[_i]^2 - Δ^2/m.Δtime_point[_i]^3)
            ν = 1 - 4Δ/m.Δtime_point[_i] + 3Δ^2/m.Δtime_point[_i]^2
            τ = -2Δ/m.Δtime_point[_i] + 3Δ^2/m.Δtime_point[_i]^2
            return -φ * q[_i] + φ * q[_i+1] + ν * m.qdot[_i] + τ * m.qdot[_i+1]
        end
    end
end

@views function ddu2(m, t, q)
    T = length(m.time_point)
    for _i in 1:T-1
        if t <= m.time_point[_i+1]
            Δ = t - m.time_point[_i]
            φ = 6 * (1/m.Δtime_point[_i]^2 - 2Δ/m.Δtime_point[_i]^3)
            ν = -4/m.Δtime_point[_i] + 6Δ/m.Δtime_point[_i]^2
            τ = -2/m.Δtime_point[_i] + 6Δ/m.Δtime_point[_i]^2
            return -φ * q[_i] + φ * q[_i+1] + ν * m.qdot[_i] + τ * m.qdot[_i+1]
        end
    end
end

@views function dxdt!(m::Model, t, x, p, q)
    I = u(m, t, q)
    ex = exp.(x)
    ep = exp.(p)

    m.dxdt[1] = ρ * ep[4] * ex[9]   / ex[1]     - I * ep[1]
    m.dxdt[2] = I * ep[1] * ex[1]   / ex[2]     - 2ep[2]*ex[2]
    m.dxdt[3] =     ep[2] * ex[2]^2 / ex[3]     - ep[3]
    m.dxdt[4] =     ep[3] * ex[3]   / ex[4] / ρ - ep[4]
    m.dxdt[5] = ep[4] * (2ex[4] / ex[5] - 1.)
    m.dxdt[6] = ep[4] * ( ex[5] / ex[6] - 1.)
    m.dxdt[7] = ep[4] * ( ex[6] / ex[7] - 1.)
    m.dxdt[8] = ep[4] * ( ex[7] / ex[8] - 1.)
    m.dxdt[9] = ep[4] * ( ex[8] / ex[9] - 1.)
    nothing
end

@views function jacobianx!(m::Model, t, x, p, q) # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    I = u(m, t, q)
    ex = exp.(x)
    ep = exp.(p)

    m.jacobianx[1,1] = -ρ * ep[4] * ex[9] / ex[1]
    m.jacobianx[1,9] = -m.jacobianx[1,1]

    m.jacobianx[2,1] = I * ep[1] * ex[1] / ex[2]
    m.jacobianx[2,2] = -m.jacobianx[2,1] - 2ep[2] * ex[2]

    m.jacobianx[3,2] = 2ep[2] * ex[2]^2 / ex[3]
    m.jacobianx[3,3] = -m.jacobianx[3,2]/2.

    m.jacobianx[4,3] = ep[3] * ex[3] / ex[4] / ρ
    m.jacobianx[4,4] = -m.jacobianx[4,3]

    m.jacobianx[5,4] = 2ep[4] * ex[4] / ex[5]
    m.jacobianx[5,5] = -m.jacobianx[5,4]

    m.jacobianx[6,5] = ep[4] * ex[5] / ex[6]
    m.jacobianx[6,6] = -m.jacobianx[6,5]

    m.jacobianx[7,6] = ep[4] * ex[6] / ex[7]
    m.jacobianx[7,7] = -m.jacobianx[7,6]

    m.jacobianx[8,7] = ep[4] * ex[7] / ex[8]
    m.jacobianx[8,8] = -m.jacobianx[8,7]

    m.jacobianx[9,8] = ep[4] * ex[8] / ex[9]
    m.jacobianx[9,9] = -m.jacobianx[9,8]
    nothing
end

@views function jacobianp!(m::Model, t, x, p, q)
    I = u(m, t, q)
    ex = exp.(x)
    ep = exp.(p)

    m.jacobianp[1,1] = -I * ep[1]
    m.jacobianp[1,4] = ρ * ep[4] * ex[9] / ex[1]

    m.jacobianp[2,1] = I * ep[1] * ex[1] / ex[2]
    m.jacobianp[2,2] = -2ep[2] * ex[2]

    m.jacobianp[3,2] = ep[2] * ex[2]^2 / ex[3]
    m.jacobianp[3,3] = -ep[3]

    m.jacobianp[4,3] = ep[3] * ex[3] / ex[4] / ρ
    m.jacobianp[4,4] = -ep[4]

    m.jacobianp[5,4] = ep[4] * (2ex[4] / ex[5] - 1.)
    m.jacobianp[6,4] = ep[4] * ( ex[5] / ex[6] - 1.)
    m.jacobianp[7,4] = ep[4] * ( ex[6] / ex[7] - 1.)
    m.jacobianp[8,4] = ep[4] * ( ex[7] / ex[8] - 1.)
    m.jacobianp[9,4] = ep[4] * ( ex[8] / ex[9] - 1.)

    if t == m.time_point[1]
        m.jacobianp[1,5] = -ep[1] * ep[5]
        m.jacobianp[2,5] =  ep[1] * ex[1] / ex[2] * ep[5]
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
                m.jacobianp[1,_j+4] = -(ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]) * ep[1] * ep[_j+4]
                m.jacobianp[2,_j+4] =  (ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]) * ep[1] * ex[1] / ex[2] * ep[_j+4]
            end
            m.jacobianp[1,_i+4]  -= (1 - φ) * ep[1]                 * ep[_i+4]
            m.jacobianp[2,_i+4]  += (1 - φ) * ep[1] * ex[1] / ex[2] * ep[_i+4]
            m.jacobianp[1,_i+5]  -=      φ  * ep[1]                 * ep[_i+5]
            m.jacobianp[2,_i+5]  +=      φ  * ep[1] * ex[1] / ex[2] * ep[_i+5]
            break
        end

        if t == m.time_point[_i+1]
            m.jacobianp[1,_i+5] = -ep[1] * ep[_i+5]
            m.jacobianp[2,_i+5] =  ep[1] * ex[1] / ex[2] * ep[_i+5]
            break
        end
    end

    nothing
end



function jacobianq!(m::Model, time_point, t, x, p, q, qdot)
    ex = exp.(x)
    ep = exp.(p)
    eq = exp.(q)

    T = length(time_point)
    for _i in 1:T-1
        if t <= time_point[_i+1]
            Δ = t - time_point[_i]
            Δi = time_point[_i+1] - time_point[_i]
            φ = 3Δ^2/Δi^2 - 2Δ^3/Δi^3

            m.jacobianq[1,_i]   = -(1 - φ) * ep[1]                 * eq[_i]
            m.jacobianq[1,_i+1] = -     φ  * ep[1]                 * eq[_i+1]

            m.jacobianq[2,_i]   =  (1 - φ) * ep[1] * ex[1] / ex[2] * eq[_i]
            m.jacobianq[2,_i+1] =       φ  * ep[1] * ex[1] / ex[2] * eq[_i+1]
            return
        end
    end
end

function jacobianqdot!(m::Model, time_point, t, x, p, q, qdot)
    ex = exp.(x)
    ep = exp.(p)
    eqdot = exp.(qdot)

    T = length(time_point)
    for _i in 1:T-1
        if t <= time_point[_i+1]
            Δ = t - time_point[_i]
            Δi = time_point[_i+1] - time_point[_i]
            υ = Δ - 2Δ^2/Δi + Δ^3/Δi^2
            τ = -Δ^2/Δi + Δ^3/Δi^2

            m.jacobianqdot[1,_i]   = -υ * ep[1] * eqdot[_i]
            m.jacobianqdot[1,_i+1] = -τ * ep[1] * eqdot[_i+1]

            m.jacobianqdot[2,_i]   =  υ * ep[1] * ex[1] / ex[2] * eqdot[_i]
            m.jacobianqdot[2,_i+1] =  τ * ep[1] * ex[1] / ex[2] * eqdot[_i+1]
            return
        end
    end
end


@views function hessianxx!(m::Model, t, x, p)
    I = u(m, t, p[5:end])
    ex = exp.(x)
    ep = exp.(p)

    m.hessianxx[1,1,1] = m.hessianxx[1,9,9] = ρ * ep[4] * ep[9] / ep[1]
    m.hessianxx[1,1,9] = m.hessianxx[1,9,1] = -m.hessianxx[1,1,1]

    m.hessianxx[2,1,1] =                      I * ep[1] * ex[1] / ex[2]
    m.hessianxx[2,1,2] = m.hessianxx[2,2,1] = -m.hessianxx[2,1,1]
    m.hessianxx[2,2,2] =                       m.hessianxx[2,1,1] - 2ep[2] * ex[2]

    m.hessianxx[3,3,3] =                      ep[2] * ex[2]^2 / ex[3]
    m.hessianxx[3,2,3] = m.hessianxx[3,3,2] = -2. * m.hessianxx[3,3,3]
    m.hessianxx[3,2,2] =                       4. * m.hessianxx[3,3,3]

    m.hessianxx[4,3,3] = m.hessianxx[4,4,4] = ep[3] * ex[3] / ex[4] / ρ
    m.hessianxx[4,3,4] = m.hessianxx[4,4,3] = -m.hessianxx[4,3,3]

    m.hessianxx[5,4,4] = m.hessianxx[5,5,5] = 2ep[4] * ex[4] / ex[5]
    m.hessianxx[5,4,5] = m.hessianxx[5,5,4] = -m.hessianxx[5,4,4]

    m.hessianxx[6,5,5] = m.hessianxx[6,6,6] = ep[4] * ex[5] / ex[6]
    m.hessianxx[6,5,6] = m.hessianxx[6,6,5] = -m.hessianxx[6,5,5]

    m.hessianxx[7,6,6] = m.hessianxx[7,7,7] = ep[4] * ex[6] / ex[7]
    m.hessianxx[7,6,7] = m.hessianxx[7,7,6] = -m.hessianxx[7,6,6]

    m.hessianxx[8,7,7] = m.hessianxx[8,8,8] = ep[4] * ex[7] / ex[8]
    m.hessianxx[8,7,8] = m.hessianxx[8,8,7] = -m.hessianxx[8,7,7]

    m.hessianxx[9,8,8] = m.hessianxx[9,9,9] = ep[4] * ex[8] / ex[9]
    m.hessianxx[9,8,9] = m.hessianxx[9,9,8] = -m.hessianxx[9,8,8]
    nothing
end

function hessianxp!(m::Model, t, x, p)
    I = u(m, t, p[5:end])
    ex = exp.(x)
    ep = exp.(p)

    m.hessianxp[1,1,4] = -ρ * ep[4] * ep[9] / ep[1]
    m.hessianxp[1,9,4] = -m.hessianxp[1,1,4]

    m.hessianxp[2,1,1] = I * ep[1] * ex[1] / ex[2]
    m.hessianxp[2,2,1] = -m.hessianxp[2,1,1]
    m.hessianxp[2,2,2] = -2ep[2]*ex[2]

    m.hessianxp[3,3,2] = -ep[2] * ex[2]^2 / ex[3]
    m.hessianxp[3,2,2] = -2. * m.hessianxp[3,3,2]

    m.hessianxp[4,3,3] = ep[3] * ex[3] / ex[4] / ρ
    m.hessianxp[4,4,3] = -m.hessianxp[4,3,3]

    m.hessianxp[5,4,4] = 2ep[4] * ex[4] / ex[5]
    m.hessianxp[5,5,4] = -m.hessianxp[5,4,4]

    m.hessianxp[6,5,4] = ep[4] * ex[5] / ex[6]
    m.hessianxp[6,6,4] = -m.hessianxp[6,5,4]

    m.hessianxp[7,6,4] = ep[4] * ex[6] / ex[7]
    m.hessianxp[7,7,4] = -m.hessianxp[7,6,4]

    m.hessianxp[8,7,4] = ep[4] * ex[7] / ex[8]
    m.hessianxp[8,8,4] = -m.hessianxp[8,7,4]

    m.hessianxp[9,8,4] = ep[4] * ex[8] / ex[9]
    m.hessianxp[9,9,4] = -m.hessianxp[9,8,4]

    T = length(m.time_point)
    for _i in 1:T-1
        if t <= m.time_point[_i+1]
            Δ = t - m.time_point[_i]
            φ = 3Δ^2/m.Δtime_point[_i]^2 - 2Δ^3/m.Δtime_point[_i]^3
            ν = Δ - 2Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
            τ = -Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2

            for _j in 1:T
                m.hessianxp[2,1,_j+4] = (ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]) * ep[1] * ex[1] / ex[2] * ep[_j+4]
                m.hessianxp[2,2,_j+4] = -m.hessianxp[2,1,_j+4]
            end
            m.hessianxp[2,1,_i+4]  += (1 - φ) * ep[1] * ex[1] / ex[2] * ep[_i+4]
            m.hessianxp[2,2,_i+4]  -= (1 - φ) * ep[1] * ex[1] / ex[2] * ep[_i+4]
            m.hessianxp[2,1,_i+5]  +=      φ  * ep[1] * ex[1] / ex[2] * ep[_i+5]
            m.hessianxp[2,2,_i+5]  -=      φ  * ep[1] * ex[1] / ex[2] * ep[_i+5]
            return nothing
        end
    end

    nothing
end

function hessianpp!(m::Model, t, x, p)
    I = u(m, t, p[5:end])
    ex = exp.(x)
    ep = exp.(p)

    m.hessianpp[1,1,1] = -I * ep[1]
    m.hessianpp[1,4,4] = ρ * ep[4] * ex[9] / ex[1]

    m.hessianpp[2,1,1] = I * ep[1] * ex[1] / ex[2]
    m.hessianpp[2,2,2] = -2ep[2] * ex[2]

    m.hessianpp[3,2,2] = ep[2] * ex[2]^2 / ex[3]
    m.hessianpp[3,3,3] = -ep[3]

    m.hessianpp[4,3,3] = ep[3] * ex[3] / ex[4] / ρ
    m.hessianpp[4,4,4] = -ep[4]

    m.hessianpp[5,4,4] = ep[4] * (2ex[4] / ex[5] - 1.)
    m.hessianpp[6,4,4] = ep[4] * ( ex[5] / ex[6] - 1.)
    m.hessianpp[7,4,4] = ep[4] * ( ex[6] / ex[7] - 1.)
    m.hessianpp[8,4,4] = ep[4] * ( ex[7] / ex[8] - 1.)
    m.hessianpp[9,4,4] = ep[4] * ( ex[8] / ex[9] - 1.)

    T = length(m.time_point)
    for _i in 1:T-1
        if t <= m.time_point[_i+1]
            Δ = t - m.time_point[_i]
            φ = 3Δ^2/m.Δtime_point[_i]^2 - 2Δ^3/Δtime_point[_i]^3
            ν = Δ - 2Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
            τ = -Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2

            for _j in 1:T
                m.hessianpp[1,1,_j+4] = m.hessianpp[1,_j+4,1] = m.hessianpp[1,_j+4,_j+4] = -(ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]) * ep[1] * ep[_j+4]
                m.hessianpp[2,1,_j+4] = m.hessianpp[2,_j+4,1] = m.hessianpp[2,_j+4,_j+4] =  (ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]) * ep[1] * ex[1] / ex[2] * ep[_j+4]
            end
            m.hessianpp[1,1,_i+4]    -= (1 - φ) * ep[1]                 * ep[_i+4]
            m.hessianpp[1,_i+4,1]    -= (1 - φ) * ep[1]                 * ep[_i+4]
            m.hessianpp[1,_i+4,_i+4] -= (1 - φ) * ep[1]                 * ep[_i+4]

            m.hessianpp[1,1,_i+5]    -=      φ  * ep[1]                 * ep[_i+5]
            m.hessianpp[1,_i+5,1]    -=      φ  * ep[1]                 * ep[_i+5]
            m.hessianpp[1,_i+5,_i+5] -=      φ  * ep[1]                 * ep[_i+5]

            m.hessianpp[2,1,_i+4]    += (1 - φ) * ep[1] * ex[1] / ex[2] * ep[_i+4]
            m.hessianpp[2,_i+4,1]    += (1 - φ) * ep[1] * ex[1] / ex[2] * ep[_i+4]
            m.hessianpp[2,_i+4,_i+4] += (1 - φ) * ep[1] * ex[1] / ex[2] * ep[_i+4]

            m.hessianpp[2,1,_i+5]    +=      φ  * ep[1] * ex[1] / ex[2] * ep[_i+5]
            m.hessianpp[2,_i+5,1]    +=      φ  * ep[1] * ex[1] / ex[2] * ep[_i+5]
            m.hessianpp[2,_i+5,_i+5] +=      φ  * ep[1] * ex[1] / ex[2] * ep[_i+5]
            return
        end
    end

    nothing
end

# function observation(x, r)
#     obs = similar(x, 2)
#     obs[1] = r[1] + r[2] * (       x[2] + 2x[3])
#     obs[2] = r[3] + r[4] * (x[1] + x[2] + 2x[3])
#     return obs
# end

observation(x) = exp(x)

d_observation(x) = exp(x)

dd_observation(x) = exp(x)

inv_observation(z) = log(z)

function calc_qdot!(m::Model, q)
    T = length(m.time_point)
    m.rhs[1] = (q[2] - q[1]) / m.Δtime_point[1]
    for _i in 2:T-1
        m.rhs[_i] = (q[_i] - q[_i-1]) / m.Δtime_point[_i-1]^2 + (q[_i+1] - q[_i]) / m.Δtime_point[_i]^2
    end
    m.rhs[T] = (q[T] - q[T-1]) / m.Δtime_point[T-1]
    m.rhs .*= 3.

    m.qdot .= m.spline_lhs \ m.rhs
end
