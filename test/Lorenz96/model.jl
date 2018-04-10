function dxdt!(dxdt, t, x, p)
    N = length(dxdt)
    @inbounds dxdt[1]     = p[2] * (x[2]   - x[N-1]) * x[N]   + p[1] - x[1]
    @inbounds dxdt[2]     = p[2] * (x[3]   - x[N])   * x[1]   + p[1] - x[2]
    @inbounds for i in 3:N-1
        dxdt[i] = p[2] * (x[i+1] - x[i-2]) * x[i-1] + p[1] - x[i]
    end
    @inbounds dxdt[N]     = p[2] * (x[1]   - x[N-2]) * x[N-1] + p[1] - x[N]
    nothing
end

function jacobian!(jacobian, t, x, p) # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    N = size(jacobian, 1)
    @inbounds for j in 1:N, i in 1:N
        jacobian[i,j] = p[2]     * ( (mod1(i+1, N) == j) -  (mod1(i-2, N) == j)) * x[mod1(i-1, N)] + p[2]     * (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      *  (mod1(i-1, N) == j) - (i   == j)
    end
    @inbounds jacobian[:,N+1] .= 1.
    @inbounds for i in 1:N
        jacobian[i,N+2] = (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      * x[mod1(i-1, N)]
    end
    nothing
end

function hessian!(hessian, t, x, p)
    N = size(hessian, 1)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        hessian[i,j,k] = p[2]     * ( (mod1(i+1, N)==j) -  (mod1(i-2, N)==j)) *  (mod1(i-1, N)==k) + p[2]     * ( (mod1(i+1, N)==k) -  (mod1(i-2, N)==k)) *  (mod1(i-1, N)==j)
    end
    @inbounds for j in 1:N, i in 1:N
        hessian[i,j,N+2] = ( (mod1(i+1, N)==j) -  (mod1(i-2, N)==j)) * x[mod1(i-1, N)] + (x[mod1(i+1, N)]    - x[mod1(i-2, N)])    *  (mod1(i-1, N)==j)
    end
    @inbounds for k in 1:N, i in 1:N
        hessian[i,N+2,k] = hessian[i,k,N+2]
    end
    nothing
end
