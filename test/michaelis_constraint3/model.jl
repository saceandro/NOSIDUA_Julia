u(t) = 0.1 + 0.099*t

function dxdt!(m::Model, i0, i, t, x, p) # i0 = 1
    I = u(t[i])
    I0 = u(t[i0])

    @inbounds m.dxdt[1] = I / (p[1] + I) - I0 / ( p[1] + I0 ) * x[1,i] / x[1,i0]
    @inbounds m.dxdt[2] = x[1,i] / (p[2] + x[1,i]) - x[1,i0] / (p[2] + x[1,i0]) * x[2,i] / x[2,i0]
    nothing
end

function jacobian!(m::Model, i0, i, t, x, p) # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    I = u(t[i])
    I0 = u(t[i0])

    @inbounds m.jacobian[1,1] = -I0 / (p[1] + I0) / x[1,i0]
    @inbounds m.jacobian[1,3] = -I / (p[1] + I)^2 + I0 / (p[1] + I0)^2 * x[1,i] / x[1,i0]

    @inbounds m.jacobian[2,1] = p[2] / (p[2] + x[1,i])^2
    @inbounds m.jacobian[2,2] = -x[1,i0] / (p[2] + x[1,i0]) / x[2,i0]
    @inbounds m.jacobian[2,4] = -x[1,i] / (p[2] + x[1,i])^2 + x[1,i0] / (p[2] + x[1,i0])^2 * x[2,i] / x[2,i0]
    nothing
end

function jacobian0!(m::Model, i0, i, t, x, p)
    I0 = u(t[i0])

    @inbounds m.jacobian0[1,1] = I0 / (p[1] + I0) * x[1,i] / x[1,i0]^2

    @inbounds m.jacobian0[2,1] = -p[2] / (p[2] + x[1,i0])^2 * x[2,i] / x[2,i0]
    @inbounds m.jacobian0[2,2] = x[1,i0] / (p[2] + x[1,i0]) * x[2,i] / x[2,i0]^2
    nothing
end

function hessian!(m::Model, i0, i, t, x, p)
    I = u(t[i])
    I0 = u(t[i0])

    @inbounds m.hessian[1,1,3] = m.hessian[1,3,1] = I0 / (p[1] + I0)^2 / x[1,i0]
    @inbounds m.hessian[1,3,3]                    = 2. * ( I / (p[1] + I)^3 - I0 / (p[1] + I0)^3 * x[1,i] / x[1,i0] )

    @inbounds m.hessian[2,1,1]                    = -2. * p[2] / (p[2] + x[1,i])^3
    @inbounds m.hessian[2,1,4] = m.hessian[2,4,1] = (x[1,i] - p[2]) / (p[2] + x[1,i])^3
    @inbounds m.hessian[2,2,4] = m.hessian[2,4,2] = x[1,i0] / (p[2] + x[1,i0])^2 / x[2,i0]
    @inbounds m.hessian[2,4,4]                    = 2. * ( x[1,i] / (p[2] + x[1,i])^3 - x[1,i0] / (p[2] + x[1,i0])^3 * x[2,i] / x[2,i0] )
    nothing
end

function hessian0!(m::Model, i0, i, t, x, p)
    I0 = u(t[i0])

    @inbounds m.hessian0[1,1,1] = I0 / (p[1] + I0) / x[1,i0]^2
    @inbounds m.hessian0[1,3,1] = -I0 / (p[1] + I0)^2 * x[1,i] / x[1,i0]^2

    @inbounds m.hessian0[2,2,1] = -p[2] / (p[2] + x[1,i0])^2 / x[2,i0]
    @inbounds m.hessian0[2,2,2] = x[1,i0] / (p[2] + x[1,i0]) / x[2,i0]^2
    @inbounds m.hessian0[2,4,1] = (p[2] - x[1,i0]) / (p[2] + x[1,i0])^3 * x[2,i] / x[2,i0]
    @inbounds m.hessian0[2,4,2] = - x[1,i0] / (p[2] + x[1,i0])^2 * x[2,i] / x[2,i0]^2
    nothing
end

function hessian00!(m::Model, i0, i, t, x, p)
    I0 = u(t[i0])

    @inbounds m.hessian00[1,1,1]                      = -2. * I0 / (p[1] + I0) * x[1,i] / x[1,i0]^3

    @inbounds m.hessian00[2,1,1]                      = 2. * p[2] / (p[2] + x[1,i0])^3 * x[2,i] / x[2,i0]
    @inbounds m.hessian00[2,1,2] = m.hessian00[2,2,1] = p[2] / (p[2] + x[1,i0])^2 * x[2,i] / x[2,i0]^2
    @inbounds m.hessian00[2,2,2]                      = -2. * x[1,i0] / (p[2] + x[1,i0]) * x[2,i] / x[2,i0]^3
    nothing
end
