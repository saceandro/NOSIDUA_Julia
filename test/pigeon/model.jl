@views function dxdt!(m::Model{N}, t, x, p) where {N}
    # Nbirds = N ÷ 4
    m.dxdt[1] = x[3] * cos(x[4])
    m.dxdt[2] = x[3] * sin(x[4])
    m.dxdt[3] = -p[1] * (x[3] - p[2]) + p[3] * cos(atan(x[5]-x[1], x[6]-x[2]) - x[4])
    m.dxdt[4] = p[4] * tanh( (x[8] - x[4]) * p[5] ) + p[6] * sin( atan(x[5]-x[1], x[6]-x[2]) - x[4] ) * tanh( ( sqrt( (x[5]-x[1])^2 + (x[6]-x[2])^2 ) - p[7] ) * p[8] )

    m.dxdt[5] = x[7] * cos(x[8])
    m.dxdt[6] = x[7] * sin(x[8])
    m.dxdt[7] = -p[9] * (x[7] - p[10]) + p[11] * cos(atan(x[1]-x[5], x[2]-x[6]) - x[8])
    m.dxdt[8] = p[12] * tanh( (x[4] - x[8]) * p[13] ) + p[14] * sin( atan(x[1]-x[5], x[2]-x[6]) - x[8] ) * tanh( ( sqrt( (x[5]-x[1])^2 + (x[6]-x[2])^2 ) - p[15] ) * p[16] )
    nothing
end

@views function jacobianx!(m::Model, t, x, p) # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    nothing
end

@views function jacobianp!(m::Model, t, x, p)
    nothing
end

@views function hessianxx!(m::Model, t, x, p)
    nothing
end

function hessianxp!(m::Model, t, x, p)
    nothing
end

function hessianpp!(m::Model, t, x, p)
    nothing
end

function observation!(m::Model, t, x, p)
    m.observation[1] = x[1]
    m.observation[2] = x[2]
    m.observation[3] = x[5]
    m.observation[4] = x[6]
    nothing
end

function observation_jacobianx!(m::Model, t, x, p)
    nothing
end

function observation_jacobianp!(m::Model, t, x, p)
    nothing
end

function observation_jacobianr!(m::Model, t, x, p)
    nothing
end

function observation_hessianxx!(m::Model, t, x, p)
    nothing
end

function observation_hessianxr!(m::Model, t, x, p)
    nothing
end

function observation_hessianpp!(m::Model, t, x, p)
    nothing
end

function observation_hessianrr!(m::Model, t, x, p)
    nothing
end
