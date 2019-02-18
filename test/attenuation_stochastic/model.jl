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

function observation!(m::Model, t, x, r)
    m.observation[1] = r[1] + x[1]
    nothing
end

function observation_jacobianx!(m::Model, t, x, r)
    m.observation_jacobianx[1,1] = 1.
    nothing
end

function observation_jacobianr!(m::Model, t, x, r)
    m.observation_jacobianr[1,1] = 1.
    nothing
end

function observation_hessianxx!(m::Model, t, x, r)
    nothing
end

function observation_hessianxr!(m::Model, t, x, r)
    nothing
end

function observation_hessianrr!(m::Model, t, x, r)
    nothing
end
