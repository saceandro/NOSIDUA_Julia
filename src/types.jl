mutable struct Adjoint{N, L, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix, C<:AbstractArray}
    dt::T
    steps::Int
    obs_variance::T
    obs::B
    x::B # dim(x) = N * (steps+1)
    p::A
    dx::B
    dp::A
    λ::B
    dλ::B
    dxdt::A
    jacobian::B
    hessian::C
    Adjoint{N, L, T, A, B, C}(dt::T, steps::Int, obs_variance::T, obs::AbstractMatrix{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}, dxdt::AbstractVector{T}, jacobian::AbstractMatrix{T}, hessian::AbstractArray{T,3}) where {N,L,T,A,B,C} = new{N,L,T,A,B,C}(dt, steps, obs_variance, obs, x, p, dx, dp, λ, dλ, dxdt, jacobian, hessian)
end

function Adjoint(dt::T, steps::Int, obs_variance::T, obs::AbstractMatrix{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}) where {T<:AbstractFloat}
    xdim = size(x,1)
    θdim = xdim + length(p)
    dxdt = similar(x, xdim)
    jacobian = zeros(T, xdim, θdim)
    hessian = zeros(T, xdim, θdim, θdim)
    Adjoint{xdim, θdim, T, typeof(p), typeof(x), typeof(hessian)}(dt, steps, obs_variance, obs, x, p, dx, dp, λ, dλ, dxdt, jacobian, hessian)
end

function Adjoint(dt::T, steps::Int, obs_variance::T, obs::AbstractMatrix{T}, x0::AbstractVector{T}, p::AbstractVector{T}, dx0::AbstractVector{T}, dp::AbstractVector{T}) where {T<:AbstractFloat}
    xdim = length(x0)
    θdim = xdim + length(p)
    x = similar(x0, xdim, steps+1)
    dx = similar(dx0, xdim, steps+1)
    λ = zeros(T, θdim, steps+1)
    dλ = zeros(T, θdim, steps+1)
    Adjoint(dt, steps, obs_variance, obs, x, p, dx, dp, λ, dλ)
end
