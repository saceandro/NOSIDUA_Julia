u(t) = 0.01 + 0.041625*t

function dxdt!(m::Model, t, x, p)
    I0 = u(0.)
    I = u(t)

    @inbounds m.dxdt[1] = p[2] * ( I / (p[1] + I) - I0 / (p[1] + I0) * x[1] )
    @inbounds m.dxdt[2] = p[4] * ( x[1] / (p[3] + x[1]) - x[2] / (p[3] + 1) )
    nothing
end

function jacobian!(m::Model, t, x, p) # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    I0 = u(0.)
    I = u(t)

    @inbounds m.jacobian[1,1] = -p[2] * I0 / (p[1] + I0)
    @inbounds m.jacobian[1,3] = p[2] * ( -I/(p[1] + I)^2 + I0/(p[1] + I0)^2 * x[1] )
    @inbounds m.jacobian[1,4] = I / (p[1] + I) - I0 / (p[1] + I0) * x[1]

    @inbounds m.jacobian[2,1] = p[3] * p[4] / (p[3] + x[1])^2
    @inbounds m.jacobian[2,2] = -p[4] / (p[3] + 1.)
    @inbounds m.jacobian[2,5] = p[4] * ( -x[1] / (p[3] + x[1])^2 + x[2] / (p[3] + 1.)^2 )
    @inbounds m.jacobian[2,6] = x[1] / (p[3] + x[1]) - x[2] / (p[3] + 1.)
    nothing
end

function hessian!(m::Model, t, x, p)
    I0 = u(0.)
    I = u(t)

    @inbounds m.hessian[1,1,3] = m.hessian[1,3,1] = p[2] * I0 / (p[1] + I0)^2
    @inbounds m.hessian[1,1,4] = m.hessian[1,4,1] = -I0 / (p[1] + I0)
    @inbounds m.hessian[1,3,3]                    = 2. * p[2] * (I / (p[1] + I)^3 - I0 / (p[1] + I0)^3 * x[1])
    @inbounds m.hessian[1,3,4] = m.hessian[1,4,3] = -I / (p[1] + I)^2 + I0 / (p[1] + I0)^2 * x[1]

    @inbounds m.hessian[2,1,1]                    = -2. * p[3] * p[4] / (p[3] + x[1])^3
    @inbounds m.hessian[2,1,5] = m.hessian[2,5,1] = p[4] * (x[1] - p[3]) / (x[1] + p[3])^3
    @inbounds m.hessian[2,1,6] = m.hessian[2,6,1] = p[3] / (p[3] + x[1])^2
    @inbounds m.hessian[2,2,5] = m.hessian[2,5,2] = p[4] / (p[3] + 1.)^2
    @inbounds m.hessian[2,2,6] = m.hessian[2,6,2] = -1. / (p[3] + 1.)
    @inbounds m.hessian[2,5,5]                    = 2. * p[4] * ( x[1] / (p[3] + x[1])^3 - x[2] / (p[3] + 1.)^3 )
    @inbounds m.hessian[2,5,6] = m.hessian[2,6,5] = -x[1] / (p[3] + x[1])^2 + x[2] / (p[3] + 1.)^2
    nothing
end

observation(x) = -log2(x)

d_observation(x) = -1./x/log(2)

dd_observation(x) = 1./x^2/log(2)

inv_observation(z) = 2^(-z)
