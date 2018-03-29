using Missings, DataArrays, CatViews, Optim

# using StaticArrays
# const MArray3{N, M, L, T, S} = StaticArrays.MArray{Tuple{N, M, L}, T, 3, S}
#
# @generated function (::Type{MArray3{S1,S2,S3,T}})() where {S1,S2,S3,T}
#     return quote
#         $(Expr(:meta, :inline))
#         MArray3{S1, S2, S3, T, $(S1*S2*S3)}()
#     end
# end
#
# const FSVector{L, T} = AbstractVector{T}
# const FSMatrix{L, M, T} = AbstractMatrix{T}
# const FSArray3{L, M, N, T} = AbstractArray{T, 3}

# abstract type FSArray{S<:Tuple, T, N} <: AbstractArray{T, N} end
# const FSVector{N, T} = FSArray{Tuple{N}, T, 1}
# const FSMatrix{N, M, T} = FSArray{Tuple{N, M}, T, 2}

# mutable struct FSArray{T, N, S<:AbstractArray, R<:NTuple} <: AbstractArray{T, N}
#     a::S
#     FSArray{T,N,S,R}(a::AbstractArray{T}) where {T,N,S,R} = new{T,N,S,R}(a)
# end

# immutable FSArray{T, N, R<:Tuple} <: AbstractArray{T, N}
#     data::AbstractArray{T, N}
# end
# Base.size(A::FSArray) = size(A.data)
# Base.getindex(A::FSArray, I...) = getindex(A.data, I...)
# Base.setindex!(A::FSArray, v, I...) = setindex!(A.data, v, I...)

# mutable struct FSArray{T, N, R<:Tuple} <: AbstractArray{T, N}
#     data::AbstractArray{T, N}
# end
# Base.size(A::FSArray) = size(A.data)
# Base.getindex(A::FSArray, I...) = getindex(A.data, I...)
# Base.setindex!(A::FSArray, v, I...) = setindex!(A.data, v, I...)

mutable struct Adjoint{N, M, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix, C<:AbstractArray, D<:AbstractDataMatrix}
    dt::T
    steps::Int
    obs_variance::T
    obs::D
    x::B # dim(x) = N * (steps+1)
    p::A
    dx::B
    dp::A
    λ::B
    dλ::B
    dxdt::A
    jacobian::B
    hessian::C
    Adjoint{N, M, T, A, B, C, D}(dt::T, steps::Int, obs_variance::T, obs::AbstractDataMatrix{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}, dxdt::AbstractVector{T}, jacobian::AbstractMatrix{T}, hessian::AbstractArray{T,3}) where {N,M,T,A,B,C,D} = new{N,M,T,A,B,C,D}(dt, steps, obs_variance, obs, x, p, dx, dp, λ, dλ, dxdt, jacobian, hessian)
end

function Adjoint(dt::T, steps::Int, obs_variance::T, obs::AbstractDataMatrix{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}) where {T<:AbstractFloat}
    xdim = size(x,1)
    θdim = xdim + length(p)
    dxdt = similar(x, xdim)
    jacobian = similar(x, xdim, θdim)
    hessian = similar(x, xdim, θdim, θdim)
    Adjoint{xdim, θdim, T, typeof(p), typeof(x), typeof(hessian), typeof(obs)}(dt, steps, obs_variance, obs, x, p, dx, dp, λ, dλ, dxdt, jacobian, hessian)
end

function Adjoint(dt::T, steps::Int, obs_variance::T, obs::AbstractDataMatrix{T}, x0::AbstractVector{T}, p::AbstractVector{T}, dx0::AbstractVector{T}, dp::AbstractVector{T}) where {T<:AbstractFloat}
    xdim = length(x0)
    θdim = xdim + length(p)
    x = similar(x0, xdim, steps+1)
    dx = similar(dx0, xdim, steps+1)
    λ = similar(x0, θdim, steps+1)
    dλ = similar(x0, θdim, steps+1)
    Adjoint(dt, steps, obs_variance, obs, x, p, dx, dp, λ, dλ)
end

function dxdt!(dxdt::AbstractVector{T}, t::T, x::AbstractVector{T}, p::AbstractVector{T}) where {T<:AbstractFloat}
    N = length(x)
    dxdt[1]     = p[2] * (x[2]   - x[N-1]) * x[N]   + p[1] - x[1]
    dxdt[2]     = p[2] * (x[3]   - x[N])   * x[1]   + p[1] - x[2]
    @simd for i in 3:N-1
        dxdt[i] = p[2] * (x[i+1] - x[i-2]) * x[i-1] + p[1] - x[i]
    end
    dxdt[N]     = p[2] * (x[1]   - x[N-2]) * x[N-1] + p[1] - x[N]
    dxdt
end

function dxdt2!(dxdt::AbstractVector{T}, t::T, x::AbstractVector{T}, p::AbstractVector{T}) where {T<:AbstractFloat}
    N = length(x)
    dxdt[1]     = p[2] * (x[2]   - x[N-1]) * x[N]   + p[1] - x[1]
    dxdt[2]     = p[2] * (x[3]   - x[N])   * x[1]   + p[1] - x[2]
    @simd for i in 3:N-1
        dxdt[i] = p[2] * (x[i+1] - x[i-2]) * x[i-1] + p[1] - x[i]
    end
    dxdt[N]     = p[2] * (x[1]   - x[N-2]) * x[N-1] + p[1] - x[N]
    nothing
end

function jacobian!(jacobian::AbstractMatrix{T}, t::T, x::AbstractVector{T}, p::AbstractVector{T}) where {T<:AbstractFloat} # might be faster if SparseMatrixCSC is used. L=N+M.
    N, L = size(jacobian)
    for j in 1:L, i in 1:N
        jacobian[i,j]   = p[2]       * ( (mod1(i+1, N) == j) -  (mod1(i-2, N) == j)) * x[mod1(i-1, N)]
                        + p[2]       * (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      *  (mod1(i-1, N) == j)
                        + (N+2 == j) * (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      * x[mod1(i-1, N)]
                        + (N+1 == j)
                        - (i   == j)
    end
    jacobian
end

function hessian!(hessian::AbstractArray{T,3}, t::T, x::AbstractVector{T}, p::AbstractVector{T}) where {T<:AbstractFloat}
    N = length(x)
    L = size(hessian,2)
    for k in 1:L, j in 1:L, i in 1:N
        hessian[i,j,k]  = (N+2 == j) * ( (mod1(i+1, N)==k) -  (mod1(i-2, N)==k)) * x[mod1(i-1, N)]
                        + (N+2 == j) * (x[mod1(i+1, N)]    - x[mod1(i-2, N)])    *  (mod1(i-1, N)==k)
                        + (N+2 == k) * ( (mod1(i+1, N)==j) -  (mod1(i-2, N)==j)) * x[mod1(i-1, N)]
                        + p[2]       * ( (mod1(i+1, N)==j) -  (mod1(i-2, N)==j)) *  (mod1(i-1, N)==k)
                        + (N+2 == k) * (x[mod1(i+1, N)]    - x[mod1(i-2, N)])    *  (mod1(i-1, N)==j)
                        + p[2]       * ( (mod1(i+1, N)==k) -  (mod1(i-2, N)==k)) *  (mod1(i-1, N)==j)
    end
    hessian
end

# cost(x, obs, obs_variance) = mapreduce(abs2, +, skipmissing(x .- obs)) / obs_variance / 2. # assuming x[:,1] .= x0; orbit!(dxdt, t, x, p, dt); is already run

innovation_λ(  x, obs, obs_variance) = ismissing(obs) ? 0. : (x - obs)/obs_variance # element-wise innovation of λ

innovation_dλ(dx, obs, obs_variance) = ismissing(obs) ? 0. : dx/obs_variance        # element-wise innovation of dλ

next_x!(dxdt,             dt, t, x, p) = x .+ dxdt!(dxdt, t, x, p) .* dt

next_x2!(x_nxt, dxdt,             dt, t, x, p) = (dxdt2!(dxdt, t, x, p); x_nxt .= x .+ dxdt .* dt; nothing)

next_dx!(jacobian,        dt, t, x, p, dx, dp) = dx .+ jacobian!(jacobian, t, x, p) * Catview(dx, dp) .* dt

@views prev_λ!(jacobian,  dt, t, x, p,         λ,     obs, obs_variance) = λ .+ jacobian!(jacobian, t, x, p)' * λ[1:length(x)] .* dt .+ innovation_λ.(x, obs, obs_variance)

@views prev_dλ!(hessian,  dt, t, x, p, dx, dp, λ, dλ, obs, obs_variance) = prev_λ!(jacobian, t, x, p, dλ) .+ (hessian!(hessian, t, x, p) * CatView(dx, dp))' * λ[1:length(x)] .* dt .+ innovation_dλ.(dx, obs, obs_variance)


@views function orbit!(a)
    for _i in 1:a.steps
        a.x[:,_i+1] .= next_x!(a.dxdt, a.dt, a.dt*(_i-1), a.x[:,_i], a.p)
    end
    nothing
end

@views function orbit2!(a)
    for _i in 1:a.steps
        next_x2!(a.x[:,_i+1], a.dxdt, a.dt, a.dt*(_i-1), a.x[:,_i], a.p)
    end
    nothing
end

@views function neighboring!(a)
    for _i in 1:a.steps
        a.dx[:,_i+1] .= next_dx!(a.jacobian, a.dt, a.dt*(_i-1), a.x[:,_i], a.p, a.dx[:,_i], a.dp)
    end
    nothing
end

@views function gradient!(a) # assuming x[:,1] .= x0; p .= p; orbit!(dxdt, t, x, p, dt); is already run. λ[:,1] is the gradient.
    a.λ[:,end] .= innovation_λ(a.x[:,end], a.obs[:,end], a.obs_variance)
    for _i in a.steps:-1:1
        a.λ[:,_i] .= prev_λ!(a.jacobian,     a.dt, a.dt*(_i-1), a.x[:,_i], a.p,                   a.λ[:,_i+1],              a.obs[:,_i], a.obs_variance) # fix me! a.dt*_i?
    end
    nothing
end

@views function hessian_vector_product!(a) # assuming dx[:,1] .= dx0; dp .= dp; neighboring!(jacobian, dt, t, x, p, dx, dp); is already run. dλ[:,1] is the hessian_vector_product.
    a.dλ[:,end] .= innovation_dλ(a.dx[:,end], a.obs[:,end], a.obs_variance)
    for _i in a.steps:-1:1
        a.dλ[:,_i] .= prev_dλ(a.hessian,     a.dt, a.dt*(_i-1), a.x[:,_i], a.p, a.dx[:,_i], a.dp, a.λ[:,_i+1], a.dλ[:,_i+1], a.obs[:,_i], a.obs_variance) # fix me! a.dt*_i?
    end
    nothing
end

cost(a) = mapreduce(abs2, +, skipmissing(a.x .- a.obs)) / a.obs_variance / 2. # assuming x[:,1] .= x0; orbit!(dxdt, t, x, p, dt); is already run

ret_N(a::Adjoint{N}) where {N} = N
ret_N_M(a::Adjoint{N,M}) where{N,M} = (N,M)

@views function calculate_common!(θ, last_θ, a::Adjoint{N}) where {N} # buffering x and p do avoid recalculation of orbit between f! and g!
    if θ != last_θ
        copy!(last_θ, θ)
        copy!(a.x[:,1], θ[1:N])
        copy!(a.p, θ[N+1:end])
        orbit!(a)
    end
end

function f!(θ, a, last_θ)
    calculate_common!(θ, last_θ, a)
    cost(a.x, a.obs, a.obs_variance)
end

@views function g!(θ, ∇θ, a, last_θ)
    calculate_common!(θ, last_θ, a)
    gradient!(a)
    copy!(∇θ, a.λ[:,1])
    nothing
end

function minimize!(initial_θ, a)
    last_θ = similar(initial_θ)
    df = TwiceDifferentiable(θ -> f!(θ, a, last_θ),
                             (∇θ, θ) -> g!(θ, ∇θ, a, last_θ))
    optimize(df, initial_θ)
end


@views function covariance!(hessian, covariance, variance, stddev, a::Adjoint{N,M}) where {N,M}
    fill!(a.dx[:,1], 0.)
    fill!(a.dp, 0.)
    for i in 1:N
        a.dx[i,1] = 1.
        neighboring!(a)
        hessian_vector_product!(a)
        copy!(hess[:,i], a.dλ[:,1])
        a.dx[i,1] = 0.
    end
    for i in 1:M
        dp[i] = 1.
        neighboring!(a)
        hessian_vector_product!(a)
        copy!(hess[:,N+i], dλ[:,1])
        dp[i] = 0.
    end
    covariance .= inv(hess)
    variance .= diag(covariance)
    stddev .= sqrt.(variance)
end
