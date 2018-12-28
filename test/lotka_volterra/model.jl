@views function dxdt!(m::Model, t, x, p)
    ex = exp.(x)
    ep = exp.(p)

    m.dxdt[1] = ep[1] - ex[2]
    m.dxdt[2] = ex[1] - ep[2]
    nothing
end

@views function jacobianx!(m::Model, t, x, p) # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    ex = exp.(x)

    m.jacobianx[1,2] = -ex[2]
    m.jacobianx[2,1] =  ex[1]
    nothing
end

@views function jacobianp!(m::Model, t, x, p)
    ep = exp.(p)

    m.jacobianp[1,1] =  ep[1]
    m.jacobianp[2,2] = -ep[2]
    nothing
end

@views function hessianxx!(m::Model, t, x, p)
    ex = exp.(x)

    m.hessianxx[1,2,2] = -ex[2]
    m.hessianxx[2,1,1] =  ex[1]
    nothing
end

function hessianxp!(m::Model, t, x, p)
    nothing
end

function hessianpp!(m::Model, t, x, p)
    ep = exp.(p)

    m.hessianpp[1,1,1] =  ep[1]
    m.hessianpp[2,2,2] = -ep[2]
    nothing
end

function observation!(m::Model, t, x, p)
    ex = exp.(x)

    m.observation[1] = p[3] * ex[1]
    nothing
end

function observation_jacobianx!(m::Model, t, x, p)
    ex = exp.(x)
    m.observation_jacobianx[1,1] = p[3] * ex[1]
    nothing
end

function observation_jacobianp!(m::Model, t, x, p)
    nothing
end

function observation_jacobianr!(m::Model, t, x, p)
    ex = exp.(x)
    m.observation_jacobianr[1,1] = ex[1]
    nothing
end

function observation_hessianxx!(m::Model, t, x, p)
    ex = exp.(x)
    m.observation_hessianxx[1,1,1] = p[3] * ex[1]
    nothing
end

function observation_hessianxr!(m::Model, t, x, p)
    ex = exp.(x)
    m.observation_hessianxr[1,1,1] = ex[1]
    nothing
end

function observation_hessianpp!(m::Model, t, x, p)
    nothing
end

function observation_hessianrr!(m::Model, t, x, p)
    nothing
end
