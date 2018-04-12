mutable struct Adjoint{N, L, S, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix, C<:AbstractArray, D<:AbstractArray, F<:Function, G<:Function, H<:Function}
    dt::T
    t::A
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
    jacobians::C
    hessians::D
    dxdt!::F
    jacobian!::G
    hessian!::H
    Adjoint{N, L, S, T, A, B, C, D, F, G, H}(dt::T, t::AbstractVector{T}, obs_variance::T, obs::AbstractMatrix{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}, dxdt::AbstractVector{T}, jacobian::AbstractMatrix{T}, jacobians::AbstractArray{T,3}, hessians::AbstractArray{T,4}, dxdt!::F, jacobian!::G, hessian!::H) where {N,L,S,T,A,B,C,D,F,G,H} = new{N,L,S,T,A,B,C,D,F,G,H}(dt, t, obs_variance, obs, x, p, dx, dp, λ, dλ, dxdt, jacobian, jacobians, hessians, dxdt!, jacobian!, hessian!)
end

function Adjoint(dt::T, obs_variance::T, obs::AbstractMatrix{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}, dxdt!::F, jacobian!::G, hessian!::H) where {T<:AbstractFloat, F<:Function, G<:Function, H<:Function}
    xdim, steps = size(x)
    θdim = xdim + length(p)
    t = collect(0.:dt:dt*(steps-1))
    dxdt = similar(x, xdim)
    jacobian = zeros(T, xdim, θdim)
    jacobians = zeros(T, xdim, θdim, steps)
    hessians = zeros(T, xdim, θdim, θdim, steps)
    Adjoint{xdim, θdim, steps-1, T, typeof(p), typeof(x), typeof(jacobians), typeof(hessians), typeof(dxdt!), typeof(jacobian!), typeof(hessian!)}(dt, t, obs_variance, obs, x, p, dx, dp, λ, dλ, dxdt, jacobian, jacobians, hessians, dxdt!, jacobian!, hessian!)
end

function Adjoint(dt::T, obs_variance::T, obs::AbstractMatrix{T}, x0::AbstractVector{T}, p::AbstractVector{T}, dx0::AbstractVector{T}, dp::AbstractVector{T}, dxdt!::F, jacobian!::G, hessian!::H) where {T<:AbstractFloat, F<:Function, G<:Function, H<:Function}
    xdim, steps = size(obs)
    θdim = xdim + length(p)
    x = similar(x0, xdim, steps)
    dx = similar(dx0, xdim, steps)
    λ = zeros(T, θdim, steps)
    dλ = zeros(T, θdim, steps)
    Adjoint(dt, obs_variance, obs, x, p, dx, dp, λ, dλ, dxdt!, jacobian!, hessian!)
end
