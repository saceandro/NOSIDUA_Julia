u(t) = 0.1 + 0.099*t
# u(t) = 1. + sin(t)

function dxdt!(m::Model, i0, i, t, x, p) # i0 = 1
    # println("t[i0]:\t", t[i0])
    # println("t[i]:\t", t[i])
    I = u(t[i])
    I0 = u(t[i0])
    k1 = I0 / (1. + I0) / x[1,i0]
    @inbounds m.dxdt[1] = -k1 * x[1,i] + I / (1. + I)
    @inbounds m.dxdt[2] = p[2] * (-x[1,i0] / (p[1] + x[1,i0]) * x[2,i] / x[2,i0] + x[1,i] / (p[1] + x[1,i]) )
    nothing
end

function jacobian!(m::Model, i0, i, t, x, p) # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    # print("t[i0]:\t", t[i0])
    I0 = u(t[i0])
    k1 = I0 / (1. + I0) / x[1,i0]
    k2 = p[2] * x[1,i0] / (p[1] + x[1,i0]) / x[2,i0]

    @inbounds m.jacobian[1,1] = -k1

    @inbounds m.jacobian[2,1] = p[1] * p[2] / (p[1] + x[1,i])^2
    @inbounds m.jacobian[2,2] = -k2
    @inbounds m.jacobian[2,3] = p[2] * (x[1,i0] / (p[1] + x[1,i0])^2 * x[2,i] / x[2,i0] - x[1,i] / (p[1] + x[1,i])^2)
    @inbounds m.jacobian[2,4] = -x[1,i0] / (p[1] + x[1,i0]) * x[2,i] / x[2,i0] + x[1,i] / (p[1] + x[1,i])
    nothing
end

function jacobian0!(m::Model, i0, i, t, x, p)
    # print("t[i0]:\t", t[i0])
    I0 = u(t[i0])

    @inbounds m.jacobian0[1,1] = I0 / (1. + I0) / x[1,i0]^2 * x[1,i]

    @inbounds m.jacobian0[2,1] = -p[1] * p[2] / (p[1] + x[1,i0])^2 / x[2,i0] * x[2,i]
    @inbounds m.jacobian0[2,2] = p[2] * x[1,i0] / (p[1] + x[1,i0]) / x[2,i0]^2 * x[2,i]
    nothing
end

function hessian!(m::Model, i0, i, t, x, p)
    m.hessian[2,1,1]                    = -2. * p[1] * p[2] / (p[1] + x[1,i])^3
    m.hessian[2,1,3] = m.hessian[2,3,1] = -p[2] * (p[1] - x[1,i]) / (p[1] + x[1,i])^3
    m.hessian[2,1,4] = m.hessian[2,4,1] = p[1] / (p[1] + x[1,i])^2

    m.hessian[2,2,3] = m.hessian[2,3,2] = p[2] * x[1,i0] / (p[1] + x[1,i0])^2 / x[2,i0]
    m.hessian[2,2,4] = m.hessian[2,4,2] = -x[1,i0] / (p[1] + x[1,i0]) / x[2,i0]

    m.hessian[2,3,3]                    = p[2] * ( -2. / (p[1] + x[1,i0])^3 /x[2,i0] * x[2,i] + 3. * x[1,i] / (p[1] + x[1,i])^3 )
    m.hessian[2,3,4] = m.hessian[2,4,3] = x[1,i0] / (p[1] + x[1,i0])^2 / x[2,i0] * x[2,i] - x[1,i] / (p[1] + x[1,i])^2
    nothing
end

function hessian0!(m::Model, i0, i, t, x, p)
    I0 = u(t[i0])

    m.hessian0[1,1,1] = I0 / (1. + I0) / x[1,i0]^2


    m.hessian0[2,2,1] = -p[1] * p[2] / (p[1] + x[1,i0])^2 / x[2,i0]
    m.hessian0[2,2,2] = p[2] * x[1,i0] / (p[1] + x[1,i0]) / x[2,i0]^2

    m.hessian0[2,3,1] = p[2] * (p[1] - x[1,i0]) / (p[1] + x[1,i0])^3 / x[2,i0] * x[2,i]
    m.hessian0[2,3,2] = -p[2] * x[1,i0] / (p[1] + x[1,i0])^2 / x[2,i0]^2 * x[2,i]

    m.hessian0[2,4,1] = -p[1] / (p[1] + x[1,i0])^2 / x[2,i0] * x[2,i]
    m.hessian0[2,4,2] = x[1,i0] / (p[1] + x[1,i0]) / x[2,i0]^2 * x[2,i]
    nothing
end

function hessian00!(m::Model, i0, i, t, x, p)
    I0 = u(t[i0])

    m.hessian00[1,1,1]                      = -2. * I0 / (1. + I0) / x[1,i0]^3 * x[1,i]

    m.hessian00[2,1,1]                      = 2. * p[1] * p[2] / (p[1] + x[1,i0])^2 / x[2,i0]^2 * x[2,i]
    m.hessian00[2,1,2] = m.hessian00[2,2,1] = p[1] * p[2] / (p[1] + x[1,i0])^2 / x[2,i0]^2 * x[2,i]
    m.hessian00[2,2,2]                      = -2. * p[2] * x[1,i0] / (p[1] + x[1,i0]) / x[2,i0]^3 * x[2,i]
    nothing
end
