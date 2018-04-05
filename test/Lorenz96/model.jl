function dxdt!(a::Adjoint{N}, t, x) where {N}
    a.dxdt[1]     = a.p[2] * (x[2]   - x[N-1]) * x[N]   + a.p[1] - x[1]
    a.dxdt[2]     = a.p[2] * (x[3]   - x[N])   * x[1]   + a.p[1] - x[2]
    @simd for i in 3:N-1
        a.dxdt[i] = a.p[2] * (x[i+1] - x[i-2]) * x[i-1] + a.p[1] - x[i]
    end
    a.dxdt[N]     = a.p[2] * (x[1]   - x[N-2]) * x[N-1] + a.p[1] - x[N]
    nothing
end

function jacobian!(a::Adjoint{N}, t, x) where {N} # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    for j in 1:N, i in 1:N
        a.jacobian[i,j] = a.p[2]     * ( (mod1(i+1, N) == j) -  (mod1(i-2, N) == j)) * x[mod1(i-1, N)] + a.p[2]     * (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      *  (mod1(i-1, N) == j) - (i   == j)
    end
    a.jacobian[:,N+1] .= 1.
    @simd for i in 1:N
        a.jacobian[i,N+2] = (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      * x[mod1(i-1, N)]
    end
    nothing
end

function hessian!(a::Adjoint{N}, t, x) where {N}
    for k in 1:N, j in 1:N, i in 1:N
        a.hessian[i,j,k] = a.p[2]     * ( (mod1(i+1, N)==j) -  (mod1(i-2, N)==j)) *  (mod1(i-1, N)==k) + a.p[2]     * ( (mod1(i+1, N)==k) -  (mod1(i-2, N)==k)) *  (mod1(i-1, N)==j)
    end
    for j in 1:N, i in 1:N
        a.hessian[i,j,N+2] = ( (mod1(i+1, N)==j) -  (mod1(i-2, N)==j)) * x[mod1(i-1, N)] + (x[mod1(i+1, N)]    - x[mod1(i-2, N)])    *  (mod1(i-1, N)==j)
    end
    for k in 1:N, i in 1:N
        a.hessian[i,N+2,k] = a.hessian[i,k,N+2]
    end
    nothing
end
