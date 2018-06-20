mutable struct Model{N, L, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix, C<:AbstractArray, F<:Function, G<:Function, H<:Function, I<:Function, J<:Function, K<:Function} #<: AbstractModel{N, L, T, A, B, C, F, G, H}
    dxdt::A
    jacobian::B
    jacobian0::B
    hessian::C
    hessian0::C
    hessian00::C
    dxdt!::F
    jacobian!::G
    jacobian0!::H
    hessian!::I
    hessian0!::J
    hessian00!::K
    Model{N, L, T, A, B, C, F, G, H, I, J, K}(dxdt::AbstractVector{T}, jacobian::AbstractMatrix{T}, jacobian0::AbstractMatrix{T}, hessian::AbstractArray{T,3}, hessian0::AbstractArray{T,3}, hessian00::AbstractArray{T,3}, dxdt!::F, jacobian!::G, jacobian0!::H, hessian!::I, hessian0!::J, hessian00!::K) where {N,L,T,A,B,C,F,G,H,I,J,K} = new{N,L,T,A,B,C,F,G,H,I,J,K}(dxdt, jacobian, jacobian0, hessian, hessian0, hessian00, dxdt!, jacobian!, jacobian0!, hessian!, hessian0!, hessian00!)
end

function Model(t::Type{T}, N, L, dxdt!::F, jacobian!::G, jacobian0!::H, hessian!::I, hessian0!::J, hessian00!::K) where {T<:AbstractFloat, F<:Function, G<:Function, H<:Function, I<:Function, J<:Function, K<:Function}
    dxdt = Array{t}(N)
    jacobian = zeros(t, N, L)
    jacobian0 = zeros(t, N, N)
    hessian = zeros(t, N, L, L)
    hessian0 = zeros(t, N, L, N)
    hessian00 = zeros(t, N, N, N)
    Model{N, L, t, typeof(dxdt), typeof(jacobian), typeof(hessian), typeof(dxdt!), typeof(jacobian!), typeof(jacobian0!), typeof(hessian!), typeof(hessian0!), typeof(hessian00!)}(dxdt, jacobian, jacobian0, hessian, hessian0, hessian00, dxdt!, jacobian!, jacobian0!, hessian!, hessian0!, hessian00!)
end


mutable struct Adjoint{N, L, K, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix, C<:AbstractArray, D<:AbstractVector, E<:AbstractMatrix}
    dt::T
    steps::Int
    t::A
    obs_variance::A
    obs::C
    obs_mean::B
    obs_filterd_mean::B
    obs_filterd_var::A
    finite::E
    Nobs::D
    pseudo_obs_TSS::A
    x::B # dim(x) = N * (steps+1)
    p::A
    dx::B
    dp::A
    λ::B
    dλ::B
    Adjoint{N, L, K, T, A, B, C, D, E}(dt::T, steps::Int, t::AbstractVector{T}, obs_variance::AbstractVector{T}, obs::AbstractArray{T,3}, obs_mean::AbstractMatrix{T}, obs_filterd_mean::AbstractMatrix{T}, obs_filterd_var::AbstractVector{T}, finite::AbstractMatrix{Bool}, Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}) where {N,L,K,T,A,B,C,D,E} = new{N,L,K,T,A,B,C,D,E}(dt, steps, t, obs_variance, obs, obs_mean, obs_filterd_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ)
end

@views function Adjoint(dt::T, obs_variance::AbstractVector{T}, obs::AbstractArray{T,3}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}) where {T<:AbstractFloat}
    xdim, steps, replicates = size(obs)
    θdim = xdim + length(p)
    t = collect(0.:dt:dt*(steps-1))
    Nobs = [pseudo_Nobs[_i] + count(isfinite.(obs[_i,:,:])) for _i in 1:xdim]
    finite = Matrix{Bool}(xdim, steps)
    all!(finite, isfinite.(obs))
    obs_filterd_mean = reshape(mean(obs, 3)[finite], xdim, :)
    obs_filterd_var = reshape(sum(var(obs, 3; corrected=false)[finite], 2), xdim)
    Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(obs), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs, obs_filterd_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ)
end

function Adjoint(dt::T, obs::AbstractArray{T,3}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, x0::AbstractVector{T}, p::AbstractVector{T}, dx0::AbstractVector{T}, dp::AbstractVector{T}) where {T<:AbstractFloat}
    xdim, steps, replicates = size(obs)
    θdim = xdim + length(p)
    t = collect(0.:dt:dt*(steps-1))
    obs_variance = similar(x0)
    Nobs = [pseudo_Nobs[_i] + count(isfinite.(obs[_i,:,:])) for _i in 1:xdim]
    finite = Matrix{Bool}(xdim, steps)
    all!(finite, isfinite.(obs))
    obs_filterd_mean = mean(obs, 3)[finite]
    obs_filterd_var = var(obs, 3; corrected=false)[finite]
    x = similar(x0, xdim, steps)
    @views copy!(x[:,1], x0)
    dx = similar(dx0, xdim, steps)
    λ = zeros(T, θdim, steps)
    dλ = zeros(T, θdim, steps)
    Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(obs), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs, obs_filterd_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ)
end

function Adjoint(dt::T, obs::AbstractArray{T,3}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, x0::AbstractVector{T}, p::AbstractVector{T}) where {T<:AbstractFloat}
    xdim, steps, replicates = size(obs)
    θdim = xdim + length(p)
    t = collect(0.:dt:dt*(steps-1))
    obs_variance = similar(x0)
    Nobs = [pseudo_Nobs[_i] + count(isfinite.(obs[_i,:,:])) for _i in 1:xdim]
    finite = Matrix{Bool}(xdim, steps)
    all!(finite, isfinite.(obs))
    obs_filterd_mean = mean(obs, 3)[finite]
    obs_filterd_var = var(obs, 3; corrected=false)[finite]
    x = similar(x0, xdim, steps)
    @views copy!(x[:,1], x0)
    dx = similar(x0, xdim, steps)
    dp = similar(p)
    λ = zeros(T, θdim, steps)
    dλ = zeros(T, θdim, steps)
    Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(obs), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs, obs_filterd_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ)
end

function Adjoint(dt::T, total_T::T, obs_variance::AbstractVector{T}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, x0::AbstractVector{T}, p::AbstractVector{T}) where {T<:AbstractFloat}
    steps = Int(total_T/dt) + 1
    xdim = length(x0)
    θdim = xdim + length(p)
    t = collect(0.:dt:dt*(steps-1))
    obs = similar(x0, xdim, steps, 1)
    Nobs = copy(pseudo_Nobs)
    finite = Matrix{Bool}(xdim, steps)
    all!(finite, isfinite.(obs))
    obs_filterd_mean = mean(obs, 3)[finite]
    obs_filterd_var = var(obs, 3; corrected=false)[finite]
    x = similar(x0, xdim, steps)
    @views copy!(x[:,1], x0)
    dx = similar(x0, xdim, steps)
    dp = similar(p)
    λ = zeros(T, θdim, steps)
    dλ = zeros(T, θdim, steps)
    Adjoint{xdim, θdim, 1, T, typeof(p), typeof(x), typeof(obs), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs, obs_filterd_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ)
end

function Adjoint(dt::T, total_T::T, obs_variance::AbstractVector{T}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, x0::AbstractVector{T}, p::AbstractVector{T}, replicates::Int) where {T<:AbstractFloat}
    steps = Int(total_T/dt) + 1
    xdim = length(x0)
    θdim = xdim + length(p)
    t = collect(0.:dt:dt*(steps-1))
    obs = similar(x0, xdim, steps, replicates)
    Nobs = copy(pseudo_Nobs)
    finite = Matrix{Bool}(xdim, steps)
    # all!(finite, isfinite.(obs))
    # obs_mean = reshape(mean(obs, 3), xdim, steps)
    # obs_filterd_mean = reshape(reshape(mean(obs, 3), xdim, steps)[finite], xdim, :)
    # obs_filterd_var = reshape(reshape(var(obs, 3; corrected=false), xdim, steps)[finite], xdim, :)
    obs_mean = Matrix{T}(xdim, steps)
    obs_filterd_mean = Matrix{T}(xdim, 1)
    obs_filterd_var = Vector{T}(xdim)
    x = similar(x0, xdim, steps)
    @views copy!(x[:,1], x0)
    dx = similar(x0, xdim, steps)
    dp = similar(p)
    λ = zeros(T, θdim, steps)
    dλ = zeros(T, θdim, steps)
    Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(obs), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs, obs_mean, obs_filterd_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ) # fixed bug. K=replicates.
end

function Adjoint(dt::T, obs::AbstractArray{T,3}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, p::AbstractVector{T}) where {T<:AbstractFloat}
    xdim, steps, replicates = size(obs)
    θdim = xdim + length(p)
    t = collect(0.:dt:dt*(steps-1))
    obs_variance = similar(obs, xdim)
    Nobs = [pseudo_Nobs[_i] + count(isfinite.(obs[_i,:,:])) for _i in 1:xdim]
    finite = Matrix{Bool}(xdim, steps)
    all!(finite, isfinite.(obs))
    obs_filterd_mean = mean(obs, 3)[finite]
    obs_filterd_var = var(obs, 3; corrected=false)[finite]
    x = similar(obs, xdim, steps)
    dx = similar(obs, xdim, steps)
    dp = similar(p)
    λ = zeros(T, θdim, steps)
    dλ = zeros(T, θdim, steps)
    Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(obs), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs, obs_filterd_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ)
end

function Adjoint(dt::T, obs::AbstractArray{T,3}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, M::Int) where {T<:AbstractFloat}
    xdim, steps, replicates = size(obs)
    θdim = xdim + M
    t = collect(0.:dt:dt*(steps-1))
    obs_variance = similar(obs, xdim)
    Nobs = [pseudo_Nobs[_i] + count(isfinite.(obs[_i,:,:])) for _i in 1:xdim]
    finite = Matrix{Bool}(xdim, steps)
    all!(finite, isfinite.(obs))
    obs_filterd_mean = mean(obs, 3)[finite]
    obs_filterd_var = var(obs, 3; corrected=false)[finite]
    x = similar(obs, xdim, steps)
    p = similar(obs, M)
    dx = similar(x)
    dp = similar(p)
    λ = zeros(T, θdim, steps)
    dλ = zeros(T, θdim, steps)
    Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(obs), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs, obs_filterd_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ)
end


mutable struct AssimilationResults{T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix}
    θ::A
    obs_variance::A
    stddev::Nullable{A}
    covariance::Nullable{B}
    AssimilationResults{T, A, B}(θ::S, obs_variance::S, stddev::Nullable{S}, covariance::Nullable{U}) where {S<:AbstractVector{T}, U<:AbstractMatrix{T}} where {T,A,B} = new{T,A,B}(θ, obs_variance, stddev, covariance)
end

function AssimilationResults(θ::AbstractVector{T}, obs_variance::AbstractVector{T}, stddev::AbstractVector{T}, covariance::AbstractMatrix{T}) where {T<:AbstractFloat}
    AssimilationResults{T, typeof(θ), typeof(covariance)}(θ, obs_variance, Nullable(stddev), Nullable(covariance))
end

function AssimilationResults(θ::AbstractVector{T}, obs_variance::AbstractVector{T}, covariance::AbstractMatrix{T}) where {T<:AbstractFloat}
    AssimilationResults{T, typeof(θ), typeof(covariance)}(θ, obs_variance, Nullable{Vector{T}}(), Nullable(covariance))
end

function AssimilationResults(θ::AbstractVector{T}, obs_variance::AbstractVector{T}) where {T<:AbstractFloat}
    AssimilationResults{T, typeof(θ), Matrix{T}}(θ, obs_variance, Nullable{Vector{T}}(), Nullable{Matrix{T}}())
end
