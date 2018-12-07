ρ = 0.45/1.4

@views function u2(m, t, q)
    T = length(m.time_point)
    for _i in 1:T-1
        if t <= m.time_point[_i+1]
            Δ = t - m.time_point[_i]
            c = -m.eqdot[_i+1] - 2m.eqdot[_i] + 3(exp(q[_i+1]) - exp(q[_i]))/m.Δtime_point[_i]
            d =  m.eqdot[_i+1] +  m.eqdot[_i] - 2(exp(q[_i+1]) - exp(q[_i]))/m.Δtime_point[_i]

            return exp(q[_i]) + Δ * m.eqdot[_i] + Δ^2/m.Δtime_point[_i] * c + Δ^3/m.Δtime_point[_i]^2 * d
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
            return (1 - φ) * exp(q[_i]) + φ * exp(q[_i+1]) + ν * m.eqdot[_i] + τ * m.eqdot[_i+1]
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
            return -φ * exp(q[_i]) + φ * exp(q[_i+1]) + ν * m.eqdot[_i] + τ * m.eqdot[_i+1]
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
            return -φ * exp(q[_i]) + φ * exp(q[_i+1]) + ν * m.eqdot[_i] + τ * m.eqdot[_i+1]
        end
    end
end

@views function dxdt!(m::Model, t, x, p)
    I = u(m, t, p[5:end])
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

@views function jacobianx!(m::Model, t, x, p) # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    I = u(m, t, p[5:end])
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

@views function jacobianp!(m::Model, t, x, p)
    fill!(m.jacobianp, 0.)
    I = u(m, t, p[5:end])
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

    # if t == m.time_point[1]
    #     m.jacobianp[1,5] = -ep[1] * ep[5]
    #     m.jacobianp[2,5] =  ep[1] * ex[1] / ex[2] * ep[5]
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
                m.jacobianp[1,_j+4] = -(ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]) * ep[1] * ep[_j+4]
                m.jacobianp[2,_j+4] =  (ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]) * ep[1] * ex[1] / ex[2] * ep[_j+4]
            end
            m.jacobianp[1,_i+4]  -= (1 - φ) * ep[1]                 * ep[_i+4]
            m.jacobianp[2,_i+4]  += (1 - φ) * ep[1] * ex[1] / ex[2] * ep[_i+4]
            m.jacobianp[1,_i+5]  -=      φ  * ep[1]                 * ep[_i+5]
            m.jacobianp[2,_i+5]  +=      φ  * ep[1] * ex[1] / ex[2] * ep[_i+5]
            break
        end

        # if t == m.time_point[_i+1]
        #     m.jacobianp[1,_i+5] = -ep[1] * ep[_i+5]
        #     m.jacobianp[2,_i+5] =  ep[1] * ex[1] / ex[2] * ep[_i+5]
        #     break
        # end
    end

    nothing
end

@views function hessianxx!(m::Model, t, x, p)
    I = u(m, t, p[5:end])
    ex = exp.(x)
    ep = exp.(p)

    m.hessianxx[1,1,1] = m.hessianxx[1,9,9] = ρ * ep[4] * ex[9] / ex[1] # fixed bug
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
    fill!(m.hessianxp, 0.)
    I = u(m, t, p[5:end])
    ex = exp.(x)
    ep = exp.(p)

    m.hessianxp[1,1,4] = -ρ * ep[4] * ex[9] / ex[1] # fixed bug
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

    # if t == m.time_point[1]
    #     m.hessianxp[2,1,5] = ep[1] * ex[1] / ex[2] * ep[5]
    #     m.hessianxp[2,2,5] = -m.hessianxp[2,1,5]
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
                m.hessianxp[2,1,_j+4] = (ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]) * ep[1] * ex[1] / ex[2] * ep[_j+4]
                m.hessianxp[2,2,_j+4] = -m.hessianxp[2,1,_j+4]
            end
            m.hessianxp[2,1,_i+4]  += (1 - φ) * ep[1] * ex[1] / ex[2] * ep[_i+4]
            m.hessianxp[2,2,_i+4]  -= (1 - φ) * ep[1] * ex[1] / ex[2] * ep[_i+4]
            m.hessianxp[2,1,_i+5]  +=      φ  * ep[1] * ex[1] / ex[2] * ep[_i+5]
            m.hessianxp[2,2,_i+5]  -=      φ  * ep[1] * ex[1] / ex[2] * ep[_i+5]
            break
        end

        # if t == m.time_point[_i+1]
        #     m.hessianxp[2,1,_i+5] = ep[1] * ex[1] / ex[2] * ep[_i+5]
        #     m.hessianxp[2,2,_i+5] = -m.hessianxp[2,1,_i+5]
        #     break
        # end
    end

    nothing
end

function hessianpp!(m::Model, t, x, p)
    fill!(m.hessianpp, 0.)
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

    # if t == m.time_point[1]
    #     m.hessianpp[1,1,5] = m.hessianpp[1,5,1] = m.hessianpp[1,5,5] = -ep[5] * ep[1]
    #     m.hessianpp[2,1,5] = m.hessianpp[2,5,1] = m.hessianpp[2,5,5] = ep[5] * ep[1] * ex[1] / ex[2]
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
            break
        end

        # if t == m.time_point[_i+1]
        #     m.hessianpp[1,1,_i+5] = m.hessianpp[1,_i+5,1] = m.hessianpp[1,_i+5,_i+5] = -ep[_i+5] * ep[1]
        #     m.hessianpp[2,1,_i+5] = m.hessianpp[2,_i+5,1] = m.hessianpp[2,_i+5,_i+5] =  ep[_i+5] * ep[1] * ex[1] / ex[2]
        #     break
        # end
    end

    nothing
end

function observation!(m::Model, t, x, p)
    ex = exp.(x)
    m.observation[1] = p[10] +                  ex[2] + 2ex[3]
    m.observation[2] = p[11] + p[12] * (ex[1] + ex[2] + 2ex[3])
    m.observation[3] = u(m, t, p[5:9])
    nothing
end

function observation_jacobianx!(m::Model, t, x, p)
    ex = exp.(x)
    m.observation_jacobianx[1,2] =  ex[2]
    m.observation_jacobianx[1,3] = 2ex[3]

    m.observation_jacobianx[2,1] =  p[12] * ex[1]
    m.observation_jacobianx[2,2] =  p[12] * ex[2]
    m.observation_jacobianx[2,3] = 2p[12] * ex[3]
    nothing
end

function observation_jacobianp!(m::Model, t, x, p)
    fill!(m.observation_jacobianp, 0.)
    ep = exp.(p)

    T = length(m.time_point)
    # for _i in 1:T-1
    #     if m.time_point[_i] <= t && t <= m.time_point[_i+1]
    #         Δ = t - m.time_point[_i]
    #         φ = 3Δ^2/m.Δtime_point[_i]^2 - 2Δ^3/m.Δtime_point[_i]^3
    #         ν = Δ - 2Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
    #         τ = -Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
    #
    #         for _j in 1:T
    #             m.observation_jacobianp[3,_j+4] = (ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]) * ep[_j+4]
    #         end
    #         m.observation_jacobianp[3,_i+4] += (1 - φ) * ep[_i+4]
    #         m.observation_jacobianp[3,_i+5] +=      φ  * ep[_i+5]
    #         break
    #     end
    for _i in 1:T
        if t == m.time_point[_i]
            m.observation_jacobianp[3,_i+4] = ep[_i+4]
            break
        end
    end
    nothing
end

function observation_jacobianr!(m::Model, t, x, p)
    ex = exp.(x)
    m.observation_jacobianr[1,1] = 1.

    m.observation_jacobianr[2,2] = 1.
    m.observation_jacobianr[2,3] = ex[1] + ex[2] + 2ex[3]
    nothing
end

function observation_hessianxx!(m::Model, t, x, p)
    ex = exp.(x)
    m.observation_hessianxx[1,2,2] =  ex[2]
    m.observation_hessianxx[1,3,3] = 2ex[3]

    m.observation_hessianxx[2,1,1] =  p[12] * ex[1]
    m.observation_hessianxx[2,2,2] =  p[12] * ex[2]
    m.observation_hessianxx[2,3,3] = 2p[12] * ex[3]
    nothing
end

function observation_hessianxr!(m::Model, t, x, p)
    ex = exp.(x)
    m.observation_hessianxr[2,1,3] =  ex[1]
    m.observation_hessianxr[2,2,3] =  ex[2]
    m.observation_hessianxr[2,3,3] = 2ex[3]
    nothing
end

function observation_hessianpp!(m::Model, t, x, p)
    fill!(m.observation_hessianpp, 0.)
    ep = exp.(p)

    T = length(m.time_point)
    # for _i in 1:T-1
    #     if m.time_point[_i] <= t && t <= m.time_point[_i+1]
    #         Δ = t - m.time_point[_i]
    #         φ = 3Δ^2/m.Δtime_point[_i]^2 - 2Δ^3/m.Δtime_point[_i]^3
    #         ν = Δ - 2Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
    #         τ = -Δ^2/m.Δtime_point[_i] + Δ^3/m.Δtime_point[_i]^2
    #
    #         for _j in 1:T
    #             m.observation_hessianpp[3,_j+4,_j+4] = (ν * m.spline_lhs_inv_rhs[_i,_j] + τ * m.spline_lhs_inv_rhs[_i+1,_j]) * ep[_j+4]
    #         end
    #         m.observation_hessianpp[3,_i+4,_i+4] += (1 - φ) * ep[_i+4]
    #         m.observation_hessianpp[3,_i+5,_i+5] +=      φ  * ep[_i+5]
    #         break
    #     end
    for _i in 1:T
        if t == m.time_point[_i]
            m.observation_hessianpp[3,_i+4,_i+4] = ep[_i+4]
            break
        end
    end
    nothing
end

function observation_hessianrr!(m::Model, t, x, p)
    nothing
end

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
