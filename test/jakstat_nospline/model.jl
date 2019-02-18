ρ = 0.45/1.4

u(t) = 1. + sin(t)

@views function dxdt!(m::Model, t, x, p)
    I = u(t)
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
    I = u(t)
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
    I = u(t)
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
    nothing
end

@views function hessianxx!(m::Model, t, x, p)
    I = u(t)
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
    I = u(t)
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
    nothing
end

function hessianpp!(m::Model, t, x, p)
    I = u(t)
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
    nothing
end

# observation(x) = exp(x)
#
# d_observation(x) = exp(x)
#
# dd_observation(x) = exp(x)
#
# inv_observation(z) = log(z)

observation(x) = x

d_observation(x) = 1.

dd_observation(x) = 0.

inv_observation(z) = z
