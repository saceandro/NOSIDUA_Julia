@views function dxdt!(m::Model{N}, t, x, p) where {N}
    Nbirds = N ÷ 4
    for i in 1:Nbirds
        m.dxdt[i] = x[2Nbirds+i] * cos(x[3Nbirds+i])
        m.dxdt[Nbirds+i] = x[2Nbirds+i] * cos(x[3Nbirds+i])
        m.dxdt[2Nbirds+i] = -p[1] * (x[2Nbirds+i] - p[2])
        m.dxdt[3Nbirds+i] = 0.
        for j in 1:Nbirds
            if j != i
                m.dxdt[2Nbirds+i] += p[3] * cos(atan(x[j]-x[i], x[Nbirds+j]-x[Nbirds+i]) - x[3Nbirds+i])
                m.dxdt[3Nbirds+i] += p[4] * tanh( (x[3Nbirds+j] - x[3Nbirds+i]) * p[5] ) + p[6] * sin( atan(x[j]-x[i], x[Nbirds+j]-x[Nbirds+i]) - x[3Nbirds+i] ) * tanh( ( sqrt( (x[j]-x[i])^2 + (x[Nbirds+j]-x[Nbirds+i])^2 ) - p[7] ) * p[8] )
            end
        end
    end
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

function observation!(m::Model{N}, t, x, p) where {N}
    Nbirds = N ÷ 4
    m.observation .= x[1:2Nbirds]
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
