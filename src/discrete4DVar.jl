module Discrete4DVar

using CatViews, Missings, DataArrays, NLSolversBase, Optim

export Adjoint, minimize!, covariance!

mutable struct Adjoint{N, L, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix, C<:AbstractArray, D<:AbstractDataMatrix}
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
    Adjoint{N, L, T, A, B, C, D}(dt::T, steps::Int, obs_variance::T, obs::AbstractDataMatrix{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}, dxdt::AbstractVector{T}, jacobian::AbstractMatrix{T}, hessian::AbstractArray{T,3}) where {N,L,T,A,B,C,D} = new{N,L,T,A,B,C,D}(dt, steps, obs_variance, obs, x, p, dx, dp, λ, dλ, dxdt, jacobian, hessian)
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

function dxdt!(a::Adjoint{N, L, T}, t::T, x::AbstractVector{T}) where {N, L, T<:AbstractFloat}
    a.dxdt[1]     = a.p[2] * (x[2]   - x[N-1]) * x[N]   + a.p[1] - x[1]
    a.dxdt[2]     = a.p[2] * (x[3]   - x[N])   * x[1]   + a.p[1] - x[2]
    @simd for i in 3:N-1
        a.dxdt[i] = a.p[2] * (x[i+1] - x[i-2]) * x[i-1] + a.p[1] - x[i]
    end
    a.dxdt[N]     = a.p[2] * (x[1]   - x[N-2]) * x[N-1] + a.p[1] - x[N]
    nothing
end

function jacobian!(a::Adjoint{N, L, T}, t::T, x::AbstractVector{T}) where {N, L, T<:AbstractFloat} # might be faster if SparseMatrixCSC is used. L=N+M.
    for j in 1:L, i in 1:N
        a.jacobian[i,j] = a.p[2]     * ( (mod1(i+1, N) == j) -  (mod1(i-2, N) == j)) * x[mod1(i-1, N)]
                        + a.p[2]     * (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      *  (mod1(i-1, N) == j)
                        + (N+2 == j) * (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      * x[mod1(i-1, N)]
                        + (N+1 == j)
                        - (i   == j)
    end
    nothing
end

function hessian!(a::Adjoint{N, L, T}, t::T, x::AbstractVector{T}) where {N, L, T<:AbstractFloat}
    for k in 1:L, j in 1:L, i in 1:N
        a.hessian[i,j,k] = (N+2 == j) * ( (mod1(i+1, N)==k) -  (mod1(i-2, N)==k)) * x[mod1(i-1, N)]
                         + (N+2 == j) * (x[mod1(i+1, N)]    - x[mod1(i-2, N)])    *  (mod1(i-1, N)==k)
                         + (N+2 == k) * ( (mod1(i+1, N)==j) -  (mod1(i-2, N)==j)) * x[mod1(i-1, N)]
                         + a.p[2]     * ( (mod1(i+1, N)==j) -  (mod1(i-2, N)==j)) *  (mod1(i-1, N)==k)
                         + (N+2 == k) * (x[mod1(i+1, N)]    - x[mod1(i-2, N)])    *  (mod1(i-1, N)==j)
                         + a.p[2]     * ( (mod1(i+1, N)==k) -  (mod1(i-2, N)==k)) *  (mod1(i-1, N)==j)
    end
    nothing
end

# cost(x, obs, obs_variance) = mapreduce(abs2, +, skipmissing(x .- obs)) / obs_variance / 2. # assuming x[:,1] .= x0; orbit!(dxdt, t, x, p, dt); is already run

innovation_λ(  x, obs, obs_variance) = ismissing(obs) ? 0. : (x - obs)/obs_variance # element-wise innovation of λ

innovation_dλ(dx, obs, obs_variance) = ismissing(obs) ? 0. : dx/obs_variance        # element-wise innovation of dλ

next_x!(        a::Adjoint,       t, x, x_nxt)                                               = (dxdt!(    a, t, x);                                  x_nxt .=  x .+ a.dxdt                                                             .* a.dt; nothing)

next_dx!(       a::Adjoint,       t, x,        dx, dx_nxt)                                   = (jacobian!(a, t, x);                                 dx_nxt .= dx .+ a.jacobian  * CatView(dx, a.dp)                                    .* a.dt; nothing)

@views prev_λ!( a::Adjoint{N},    t, x,                   λ, λ_prev)            where {N}    = (jacobian!(a, t, x);                                 λ_prev .=  λ .+ a.jacobian' *  λ[1:N]                                              .* a.dt; nothing)

@views prev_dλ!(a::Adjoint{N, L}, t, x,        dx,        λ,       dλ, dλ_prev) where {N, L} = (hessian!( a, t, x); prev_λ!(a, t, x, dλ, dλ_prev); dλ_prev .+= reshape(reshape(a.hessian, N*L, L) * CatView(dx, a.dp), N, L)' * λ[1:N] .* a.dt; nothing)
# @views function prev_dλ!(dλ_nxt, jacobian, hessian, dt, t, x, p, dx, dp, λ, dλ)
#     hessian!(hessian, t, x, p)
#     prev_λ!(dλ_nxt, jacobian, dt, t, x, p, dλ)
#     N = length(x)
#     L = length(λ)
#     dλ_nxt .+= reshape(reshape(hessian, N*L, L) * CatView(dx, dp), N, L)' * λ[1:length(x)] .* dt
#     nothing
# end

@views function orbit!(a)
    for _i in 1:a.steps
        next_x!(a, a.dt*(_i-1), a.x[:,_i], a.x[:,_i+1])
    end
    nothing
end

@views function neighboring!(a)
    for _i in 1:a.steps
        next_dx!(a, a.dt*(_i-1), a.x[:,_i],            a.dx[:,_i], a.dx[:,_i+1])
    end
    nothing
end

@views function gradient!(a::Adjoint{N}) where {N} # assuming x[:,1] .= x0; p .= p; orbit!(dxdt, t, x, p, dt); is already run. λ[:,1] is the gradient.
    a.λ[1:N,end] .= innovation_λ.(a.x[:,end], a.obs[:,end], a.obs_variance)
    for _i in a.steps:-1:1
        prev_λ!(a, a.dt*(_i-1), a.x[:,_i],                                      a.λ[:,_i+1], a.λ[:,_i]) # fix me! a.dt*_i?
        a.λ[1:N,_i] .+= innovation_λ.(a.x[:,_i], a.obs[:,_i], a.obs_variance)
    end
    nothing
end

@views function hessian_vector_product!(a::Adjoint{N}) where {N} # assuming dx[:,1] .= dx0; dp .= dp; neighboring!(jacobian, dt, t, x, p, dx, dp); is already run. dλ[:,1] is the hessian_vector_product.
    a.dλ[1:N,end] .= innovation_dλ.(a.dx[:,end], a.obs[:,end], a.obs_variance)
    for _i in a.steps:-1:1
        prev_dλ!(a, a.dt*(_i-1), a.x[:,_i],            a.dx[:,_i],              a.λ[:,_i+1],            a.dλ[:,_i+1], a.dλ[:,_i]) # fix me! a.dt*_i?
        a.dλ[1:N,_i] .+= innovation_dλ.(a.dx[:,_i], a.obs[:,_i], a.obs_variance)
    end
    nothing
end

cost(a) = mapreduce(abs2, +, skipmissing(a.x .- a.obs)) / a.obs_variance / 2. # assuming x[:,1] .= x0; orbit!(dxdt, t, x, p, dt); is already run

# @views function calculate_common!(θ, last_θ, a::Adjoint{N}) where {N} # buffering x and p do avoid recalculation of orbit between f! and g!
#     if θ != last_θ
#         copy!(last_θ, θ)
#         copy!(a.x[:,1], θ[1:N])
#         copy!(a.p, θ[N+1:end])
#         orbit!(a)
#     end
# end
#
# function f!(θ, a, last_θ)
#     calculate_common!(θ, last_θ, a)
#     cost(a)
# end
#
# @views function g!(θ, ∇θ, a, last_θ)
#     calculate_common!(θ, last_θ, a)
#     gradient!(a)
#     copy!(∇θ, a.λ[:,1])
#     nothing
# end

function fg!(F, ∇θ, θ, a::Adjoint{N}) where {N}
    copy!(a.x[:,1], θ[1:N])
    copy!(a.p, θ[N+1:end])
    orbit!(a)
    if !(∇θ == nothing)
        gradient!(a)
        copy!(∇θ, a.λ[:,1])
    end
    if !(F == nothing)
        F = cost(a)
    end
end

# function minimize!(initial_θ, a)
#     df = NLSolversBase.OnceDifferentiable(NLSolversBase.only_fg!((F, ∇θ, θ) -> fg!(F, ∇θ, θ, a)), initial_θ)
#     Optim.optimize(df, initial_θ, BFGS())
# end

function minimize!(initial_θ, a)
    F = 0.
    df = OnceDifferentiable(θ -> fg!(F, nothing, θ, a), (∇θ, θ) -> fg!(nothing, ∇θ, θ, a), (∇θ, θ) -> fg!(F, ∇θ, θ, a), initial_θ, F)
    optimize(df, initial_θ, BFGS())
end

@views function covariance!(hessian, covariance, variance, stddev, a::Adjoint{N,L}) where {N,L}
    fill!(a.dx[:,1], 0.)
    fill!(a.dp, 0.)
    for i in 1:N
        a.dx[i,1] = 1.
        neighboring!(a)
        hessian_vector_product!(a)
        copy!(hessian[:,i], a.dλ[:,1])
        a.dx[i,1] = 0.
    end
    for i in N+1:L
        a.dp[i-N] = 1.
        neighboring!(a)
        hessian_vector_product!(a)
        copy!(hessian[:,i], a.dλ[:,1])
        a.dp[i-N] = 0.
    end
    println(hessian)
    covariance .= inv(hessian)
    variance .= diag(covariance)
    stddev .= sqrt.(variance)
end

end

import DataArrays: @data
using Discrete4DVar
a = Adjoint(0.1, 10, 1., @data(randn(5, 11)), randn(5), randn(2), randn(5), randn(2))
initial_θ = randn(7)
@code_typed minimize!(initial_θ, a)
hessian = similar(a.x, 7, 7)
covariance = similar(a.x, 7, 7)
variance = similar(a.x, 7)
stddev = similar(a.x, 7)
@code_typed covariance!(hessian, covariance, variance, stddev, a)
