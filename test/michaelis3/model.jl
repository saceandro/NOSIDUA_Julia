u(t) = 0.1 + 0.099*t
# u(t) = 1. + sin(t)

function dxdt!(m::Model, t, x, p)
    I = u(t)
    @inbounds m.dxdt[1] = -p[1] * x[1] +        I    / (1. + I   )
    @inbounds m.dxdt[2] = -p[2] * x[2] + p[4] * x[1] / (p[3] + x[1])
    nothing
end

function jacobian!(m::Model, t, x, p) # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    I = u(t)
    @inbounds m.jacobian[1,1] = -p[1]
    @inbounds m.jacobian[1,3] = -x[1]

    @inbounds m.jacobian[2,1] = p[3] * p[4] / (p[3] + x[1])^2
    @inbounds m.jacobian[2,2] = -p[2]
    @inbounds m.jacobian[2,4] = -x[2]
    @inbounds m.jacobian[2,5] = -p[4] * x[1] / (p[3] + x[1])^2
    @inbounds m.jacobian[2,6] = x[1] / (p[3] + x[1])
    nothing
end

function hessian!(m::Model, t, x, p)
    I = u(t)
    @inbounds m.hessian[1,1,3] = m.hessian[1,3,1] = -1.

    @inbounds m.hessian[2,1,1]                    = -2. * p[3] * p[4] / (p[3] + x[1])^3
    @inbounds m.hessian[2,1,5] = m.hessian[2,5,1] = p[4] * (x[1] - p[3]) / (x[1] + p[3])^3
    @inbounds m.hessian[2,1,6] = m.hessian[2,6,1] = p[3] / (x[1] + p[3])^2
    @inbounds m.hessian[2,2,4] = m.hessian[2,4,2] = -1.
    @inbounds m.hessian[2,5,5]                    = 2. * p[4] * x[1] / (x[1] + p[3])^3
    @inbounds m.hessian[2,5,6] = m.hessian[2,6,5] = -x[1] / (x[1] + p[3])^2
    nothing
end
