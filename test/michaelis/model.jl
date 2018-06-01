u(t) = 0.01 + 0.999*t
# u(t) = 1. + sin(t)

function dxdt!(m::Model, t, x, p)
    I = u(t)
    @inbounds m.dxdt[1] = -p[1] * x[1] + p[3] * I    / (p[2] + I   )
    @inbounds m.dxdt[2] = -p[4] * x[2] + p[6] * x[1] / (p[5] + x[1])
    nothing
end

function jacobian!(m::Model, t, x, p) # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    I = u(t)
    @inbounds m.jacobian[1,1] = -p[1]
    @inbounds m.jacobian[1,3] = -x[1]
    @inbounds m.jacobian[1,4] = -p[3] * I / (p[2] + I)^2
    @inbounds m.jacobian[1,5] = I / (p[2] + I)

    @inbounds m.jacobian[2,1] = p[5] * p[6] / (p[5] + x[1])^2
    @inbounds m.jacobian[2,2] = -p[4]
    @inbounds m.jacobian[2,6] = -x[2]
    @inbounds m.jacobian[2,7] = -p[6] * x[1] / (p[5] + x[1])^2
    @inbounds m.jacobian[2,8] = x[1] / (p[5] + x[1])
    nothing
end

function hessian!(m::Model, t, x, p)
    I = u(t)
    @inbounds m.hessian[1,1,3] = m.hessian[1,3,1] = -1.
    @inbounds m.hessian[1,4,4]                    = 2. * p[3] * I / (p[2] + I)^3
    @inbounds m.hessian[1,4,5] = m.hessian[1,5,4] = -I / (p[2] + I)^2

    @inbounds m.hessian[2,1,1]                    = -p[5] * p[6] / (p[5] + x[1])^3
    @inbounds m.hessian[2,1,7] = m.hessian[2,7,1] = p[6] * (x[1] - p[5]) / (x[1] + p[5])^3
    @inbounds m.hessian[2,1,8] = m.hessian[2,8,1] = p[5] / (x[1] + p[5])^3
    @inbounds m.hessian[2,2,6] = m.hessian[2,6,2] = -1.
    @inbounds m.hessian[2,7,7]                    = 2. * p[6] * x[1] / (x[1] + p[5])^3
    @inbounds m.hessian[2,7,8] = m.hessian[2,8,8] = -x[1] / (x[1] + p[5])^2
    nothing
end
