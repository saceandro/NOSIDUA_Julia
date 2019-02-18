@views function dxdt!(m::Model, t, x, p)
    m.dxdt[1] = -p[1] * x[1]
    nothing
end

@views function jacobianx!(m::Model, t, x, p) # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    m.jacobianx[1,1] = -p[1]
    nothing
end

@views function jacobianp!(m::Model, t, x, p)
    m.jacobianp[1,1] = -x[1]
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
