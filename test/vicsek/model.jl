Δt = 0.125
v = 20.62
radius = 10

@views function next_x!(m::Model{N}, t, x, x_next, p) where {N}
    B = N ÷ 4
    for i in 1:B
        x_next[     i] = x[    i] + x[2B + i] * Δt
        x_next[B  + i] = x[B + i] + x[3B + i] * Δt
        Vx = 0
        Vy = 0
        neighbors = 0
        for j in 1:B
            if j != i && (x[i] - x[j])^2 + (x[B + i] - x[B + j])^2 < radius^2
                Vx += x[2B + j]
                Vy += x[2B + j]
                neighbors += 1
            end
        end
        Vxmean = Vx / neighbors
        Vymean = Vy / neighbors
        Vmean = sqrt(Vxmean^2 + Vymean^2)
        x_next[2B + i] = v * Vxmean / Vmean + 0.2 * v * randn()
        x_next[3B + i] = v * Vymean / Vmean + 0.2 * v * randn()
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

function observation!(m::Model{N}, t, x, r) where {N}
    B = N ÷ 4
    m.observation .= x[1:2B]
    nothing
end

function observation_jacobianx!(m::Model, t, x, r)
    nothing
end

function observation_jacobianp!(m::Model, t, x, r)
    nothing
end

function observation_jacobianr!(m::Model, t, x, r)
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

function observation_hessianpp!(m::Model, t, x, r)
    nothing
end
