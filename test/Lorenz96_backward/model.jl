function dxdt!(m::Model{N}, t, x, p) where {N}
    @inbounds m.dxdt[1]     = p[2] * (x[2]   - x[N-1]) * x[N]   + p[1] - x[1]
    @inbounds m.dxdt[2]     = p[2] * (x[3]   - x[N])   * x[1]   + p[1] - x[2]
    @inbounds for i in 3:N-1
        m.dxdt[i]           = p[2] * (x[i+1] - x[i-2]) * x[i-1] + p[1] - x[i]
    end
    @inbounds m.dxdt[N]     = p[2] * (x[1]   - x[N-2]) * x[N-1] + p[1] - x[N]
    nothing
end

function jacobianx!(m::Model{N}, t, x, p) where {N} # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    @inbounds for j in 1:N, i in 1:N
        m.jacobianx[i,j] = p[2]     * ( (mod1(i+1, N) == j) -  (mod1(i-2, N) == j)) * x[mod1(i-1, N)] + p[2]     * (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      *  (mod1(i-1, N) == j) - (i   == j)
    end
    nothing
end

function jacobianp!(m::Model{N}, t, x, p) where {N}
    @inbounds m.jacobianp[:,1] .= 1.
    @inbounds for i in 1:N
        m.jacobianp[i,2] = (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      * x[mod1(i-1, N)]
    end
    nothing
end

function hessianxx!(m::Model{N}, t, x, p) where {N}
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        m.hessianxx[i,j,k] = p[2]     * ( (mod1(i+1, N)==j) -  (mod1(i-2, N)==j)) *  (mod1(i-1, N)==k) + p[2]     * ( (mod1(i+1, N)==k) -  (mod1(i-2, N)==k)) *  (mod1(i-1, N)==j)
    end
    nothing
end

function hessianxp!(m::Model{N}, t, x, p) where {N}
    @inbounds for j in 1:N, i in 1:N
        m.hessianxp[i,j,2] = ( (mod1(i+1, N)==j) -  (mod1(i-2, N)==j)) * x[mod1(i-1, N)] + (x[mod1(i+1, N)]    - x[mod1(i-2, N)])    *  (mod1(i-1, N)==j)
    end
    nothing
end

function hessianpp!(m::Model, t, x, p)
    nothing
end

observation(x) = x

d_observation(x) = 1.

dd_observation(x) = 0.

inv_observation(z) = z
