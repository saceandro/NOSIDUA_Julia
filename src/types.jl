mutable struct Model{N, L, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix, C<:AbstractArray, F<:Function, G<:Function, H<:Function} #<: AbstractModel{N, L, T, A, B, C, F, G, H}
    dxdt::A
    jacobian::B
    hessian::C
    dxdt!::F
    jacobian!::G
    hessian!::H
    Model{N, L, T, A, B, C, F, G, H}(dxdt::AbstractVector{T}, jacobian::AbstractMatrix{T}, hessian::AbstractArray{T,3}, dxdt!::F, jacobian!::G, hessian!::H) where {N,L,T,A,B,C,F,G,H} = new{N,L,T,A,B,C,F,G,H}(dxdt, jacobian, hessian, dxdt!, jacobian!, hessian!)
end

function Model(t::Type{T}, N, L, dxdt!::F, jacobian!::G, hessian!::H) where {T<:AbstractFloat, F<:Function, G<:Function, H<:Function}
    dxdt = Array{t}(N)
    jacobian = zeros(t, N, L)
    hessian = zeros(t, N, L, L)
    Model{N, L, t, typeof(dxdt), typeof(jacobian), typeof(hessian), typeof(dxdt!), typeof(jacobian!), typeof(hessian!)}(dxdt, jacobian, hessian, dxdt!, jacobian!, hessian!)
end


mutable struct Adjoint{N, L, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix}
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
    Adjoint{N, L, T, A, B}(dt::T, steps::Int, obs_variance::T, obs::AbstractMatrix{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}) where {N,L,T,A,B} = new{N,L,T,A,B}(dt, steps, obs_variance, obs, x, p, dx, dp, λ, dλ)
end

function Adjoint(dt::T, obs_variance::T, obs::AbstractMatrix{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}) where {T<:AbstractFloat}
    xdim, steps = size(x)
    θdim = xdim + length(p)
    Adjoint{xdim, θdim, T, typeof(p), typeof(x)}(dt, steps-1, obs_variance, obs, x, p, dx, dp, λ, dλ)
end

function Adjoint(dt::T, obs_variance::T, obs::AbstractMatrix{T}, x0::AbstractVector{T}, p::AbstractVector{T}, dx0::AbstractVector{T}, dp::AbstractVector{T}) where {T<:AbstractFloat}
    xdim, steps = size(obs)
    θdim = xdim + length(p)
    x = similar(x0, xdim, steps)
    @views copy!(x[:,1], x0)
    dx = similar(dx0, xdim, steps)
    λ = zeros(T, θdim, steps)
    dλ = zeros(T, θdim, steps)
    Adjoint{xdim, θdim, T, typeof(p), typeof(x)}(dt, steps-1, obs_variance, obs, x, p, dx, dp, λ, dλ)
end

function Adjoint(dt::T, obs_variance::T, obs::AbstractMatrix{T}, x0::AbstractVector{T}, p::AbstractVector{T}) where {T<:AbstractFloat}
    xdim, steps = size(obs)
    θdim = xdim + length(p)
    x = similar(x0, xdim, steps)
    @views copy!(x[:,1], x0)
    dx = similar(x0, xdim, steps)
    dp = similar(p)
    λ = zeros(T, θdim, steps)
    dλ = zeros(T, θdim, steps)
    Adjoint{xdim, θdim, T, typeof(p), typeof(x)}(dt, steps-1, obs_variance, obs, x, p, dx, dp, λ, dλ)
end

function Adjoint(dt::T, total_T::T, obs_variance::T, x0::AbstractVector{T}, p::AbstractVector{T}) where {T<:AbstractFloat}
    steps = Int(total_T/dt) + 1
    xdim = length(x0)
    θdim = xdim + length(p)
    obs = similar(x0, xdim, steps)
    x = similar(x0, xdim, steps)
    @views copy!(x[:,1], x0)
    dx = similar(x0, xdim, steps)
    dp = similar(p)
    λ = zeros(T, θdim, steps)
    dλ = zeros(T, θdim, steps)
    Adjoint{xdim, θdim, T, typeof(p), typeof(x)}(dt, steps-1, obs_variance, obs, x, p, dx, dp, λ, dλ)
end

function Adjoint(dt::T, obs_variance::T, obs::AbstractMatrix{T}, p::AbstractVector{T}) where {T<:AbstractFloat}
    xdim, steps = size(obs)
    θdim = xdim + length(p)
    x = similar(obs, xdim, steps)
    dx = similar(obs, xdim, steps)
    dp = similar(p)
    λ = zeros(T, θdim, steps)
    dλ = zeros(T, θdim, steps)
    Adjoint{xdim, θdim, T, typeof(p), typeof(x)}(dt, steps-1, obs_variance, obs, x, p, dx, dp, λ, dλ)
end

function Adjoint(dt::T, obs_variance::T, obs::AbstractMatrix{T}, M::Int) where {T<:AbstractFloat}
    xdim, steps = size(obs)
    θdim = xdim + M
    x = similar(obs, xdim, steps)
    p = similar(obs, M)
    dx = similar(x)
    dp = similar(p)
    λ = zeros(T, θdim, steps)
    dλ = zeros(T, θdim, steps)
    Adjoint{xdim, θdim, T, typeof(p), typeof(x)}(dt, steps-1, obs_variance, obs, x, p, dx, dp, λ, dλ)
end


mutable struct AssimilationResults{T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix}
    θ::A
    stddev::Nullable{A}
    covariance::Nullable{B}
    AssimilationResults{T, A, B}(θ::S, stddev::Nullable{S}, covariance::Nullable{U}) where {S<:AbstractVector{T}, U<:AbstractMatrix{T}} where {T,A,B} = new{T,A,B}(θ, stddev, covariance)
end

function AssimilationResults(θ::AbstractVector{T}, stddev::AbstractVector{T}, covariance::AbstractMatrix{T}) where {T<:AbstractFloat}
    AssimilationResults{T, typeof(θ), typeof(covariance)}(θ, Nullable(stddev), Nullable(covariance))
end

function AssimilationResults(θ::AbstractVector{T}, covariance::AbstractMatrix{T}) where {T<:AbstractFloat}
    AssimilationResults{T, typeof(θ), typeof(covariance)}(θ, Nullable{Vector{T}}(), Nullable(covariance))
end

function AssimilationResults(θ::AbstractVector{T}) where {T<:AbstractFloat}
    AssimilationResults{T, typeof(θ), Matrix{T}}(θ, Nullable{Vector{T}}(), Nullable{Matrix{T}}())
end
