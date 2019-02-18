u(t) = 0.01 + 0.041625*t

function dxdt!(m::Model, t, x, p)
    I0 = u(0.)
    I = u(t)
    ex1 = exp(x[1])
    ex2 = exp(x[2])

    @inbounds m.dxdt[1] = p[2] * ( -I / (p[1] + I) * ex1 + I0 / (p[1] + I0) )
    @inbounds m.dxdt[2] = p[4] * ( -1. / (1. + p[3] * ex1) * ex2 + 1. / (1. + p[3]) )
    nothing
end

function jacobianx!(m::Model, t, x, p) # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    I = u(t)
    ex1 = exp(x[1])
    ex2 = exp(x[2])

    @inbounds m.jacobianx[1,1] = -p[2] * I / (I + p[1]) * ex1

    @inbounds m.jacobianx[2,1] = p[3] * p[4] / (1. + p[3] * ex1)^2 * ex1 * ex2
    @inbounds m.jacobianx[2,2] = -p[4] / (1. + p[3] * ex1) * ex2
    nothing
end

function jacobianp!(m::Model, t, x, p)
    I0 = u(0.)
    I = u(t)
    ex1 = exp(x[1])
    ex2 = exp(x[2])

    @inbounds m.jacobianp[1,1] = p[2] * ( I/(p[1] + I)^2 * ex1 - I0/(p[1] + I0)^2 )
    @inbounds m.jacobianp[1,2] = -I / (p[1] + I) * ex1 + I0 / (p[1] + I0)

    @inbounds m.jacobianp[2,3] = p[4] * ( ex1 * ex2 / (1. + p[3] * ex1)^2 - 1. / (1. + p[3])^2 )
    @inbounds m.jacobianp[2,4] = -ex2 / (1. + p[3] * ex1) + 1. / (1. + p[3])
    nothing
end

function hessianxx!(m::Model, t, x, p)
    I = u(t)
    ex1 = exp(x[1])
    ex2 = exp(x[2])

    @inbounds m.hessianxx[1,1,1]                      = -p[2] * I / (I + p[1]) * ex1

    @inbounds m.hessianxx[2,1,1]                      = p[3] * p[4] *    ex1   * ex2 * (1. - p[3] * ex1) / (1. + p[3] * ex1)^3
    @inbounds m.hessianxx[2,1,2] = m.hessianxx[2,2,1] = p[3] * p[4] *    ex1   * ex2 / (1. + p[3] * ex1)^2
    @inbounds m.hessianxx[2,2,2]                      =       -p[4]            * ex2 / (1. + p[3] * ex1)
    nothing
end

function hessianxp!(m::Model, t, x, p)
    I = u(t)
    ex1 = exp(x[1])
    ex2 = exp(x[2])

    @inbounds m.hessianxp[1,1,1] = p[2] * I / (I + p[1])^2 * ex1
    @inbounds m.hessianxp[1,1,2] = -I / (I + p[1]) * ex1

    @inbounds m.hessianxp[2,1,3] =        p[4] *    ex1   * ex2 * (1. - p[3] * ex1) / (1. + p[3] * ex1)^3
    @inbounds m.hessianxp[2,1,4] = p[3]        *    ex1   * ex2 / (1. + p[3] * ex1)^2
    @inbounds m.hessianxp[2,2,3] =        p[4] *    ex1   * ex2 / (1. + p[3] * ex1)^2
    @inbounds m.hessianxp[2,2,4] =                         -ex2 / (1. + p[3] * ex1)
    nothing
end

function hessianpp!(m::Model, t, x, p)
    I0 = u(0.)
    I = u(t)
    ex1 = exp(x[1])
    ex2 = exp(x[2])

    @inbounds m.hessianpp[1,1,1]                      = 2. * p[2] * ( -I / (I + p[1])^3 * ex1 + I0 / (I0 + p[1])^3 )
    @inbounds m.hessianpp[1,1,2] = m.hessianpp[1,2,1] = I / (I + p[1])^2 * ex1 - I0 / (I0 + p[1])^2

    @inbounds m.hessianpp[2,3,3]                      =   2. * p[4] * ( -ex1^2 * ex2 / (1. + p[3] * ex1)^3 + 1. / (1. + p[3])^3 )
    @inbounds m.hessianpp[2,3,4] = m.hessianpp[2,4,3] =                  ex1   * ex2 / (1. + p[3] * ex1)^2 - 1. / (1. + p[3])^2
    nothing
end

observation(x) = x / log(2)

d_observation(x) = 1. / log(2)

dd_observation(x) = 0.

inv_observation(z) = z * log(2)
