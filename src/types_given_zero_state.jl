mutable struct Model{N, L, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix, C<:AbstractArray, F<:Function, G<:Function, H<:Function, I<:Function, J<:Function, K<:Function, M<:Function} #<: AbstractModel{N, L, T, A, B, C, F, G, H}
    dxdt::A
    jacobian::B
    hessian::C
    dxdt!::F
    jacobian!::G
    hessian!::H
    observation::I
    d_observation::J
    dd_observation::K
    inv_observation::M
    Model{N, L, T, A, B, C, F, G, H, I, J, K, M}(dxdt::AbstractVector{T}, jacobian::AbstractMatrix{T}, hessian::AbstractArray{T,3}, dxdt!::F, jacobian!::G, hessian!::H, observation::I, d_observation::J, dd_observation::K, inv_observation::M) where {N,L,T,A,B,C,F,G,H,I,J,K,M} = new{N,L,T,A,B,C,F,G,H,I,J,K,M}(dxdt, jacobian, hessian, dxdt!, jacobian!, hessian!, observation, d_observation, dd_observation, inv_observation)
end

function Model(t::Type{T}, N, L, dxdt!::F, jacobian!::G, hessian!::H, observation::I, d_observation::J, dd_observation::K, inv_observation::M) where {T<:AbstractFloat, F<:Function, G<:Function, H<:Function, I<:Function, J<:Function, K<:Function, M<:Function}
    dxdt = Array{t}(N)
    jacobian = zeros(t, N, L)
    hessian = zeros(t, N, L, L)
    Model{N, L, t, typeof(dxdt), typeof(jacobian), typeof(hessian), typeof(dxdt!), typeof(jacobian!), typeof(hessian!), typeof(observation), typeof(d_observation), typeof(dd_observation), typeof(inv_observation)}(dxdt, jacobian, hessian, dxdt!, jacobian!, hessian!, observation, d_observation, dd_observation, inv_observation)
end


mutable struct Adjoint{N, L, K, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix, D<:AbstractVector, E<:AbstractMatrix}
    dt::T
    steps::Int
    t::A
    obs_variance::A
    obs_mean::B
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
    Adjoint{N, L, K, T, A, B, D, E}(dt::T, steps::Int, t::AbstractVector{T}, obs_variance::AbstractVector{T}, obs_mean::AbstractMatrix{T}, obs_filterd_var::AbstractVector{T}, finite::AbstractMatrix{Bool}, Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}) where {N,L,K,T,A,B,D,E} = new{N,L,K,T,A,B,D,E}(dt, steps, t, obs_variance, obs_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ)
end

# @views function Adjoint(dt::T, obs_variance::AbstractVector{T}, obs::AbstractArray{T,3}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}) where {T<:AbstractFloat}
#     xdim, steps, replicates = size(obs)
#     θdim = xdim + length(p)
#     t = collect(0.:dt:dt*(steps-1))
#     Nobs = [pseudo_Nobs[_i] + count(isfinite.(obs[_i,:,:])) for _i in 1:xdim]
#     finite = Matrix{Bool}(xdim, steps)
#     all!(finite, isfinite.(obs))
#     obs_filterd_mean = reshape(mean(obs, 3)[finite], xdim, :)
#     obs_filterd_var = reshape(sum(var(obs, 3; corrected=false)[finite], 2), xdim)
#     Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(obs), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs, obs_filterd_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ)
# end
#
# function Adjoint(dt::T, obs::AbstractArray{T,3}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, x0::AbstractVector{T}, p::AbstractVector{T}, dx0::AbstractVector{T}, dp::AbstractVector{T}) where {T<:AbstractFloat}
#     xdim, steps, replicates = size(obs)
#     θdim = xdim + length(p)
#     t = collect(0.:dt:dt*(steps-1))
#     obs_variance = similar(x0)
#     Nobs = [pseudo_Nobs[_i] + count(isfinite.(obs[_i,:,:])) for _i in 1:xdim]
#     finite = Matrix{Bool}(xdim, steps)
#     all!(finite, isfinite.(obs))
#     obs_filterd_mean = mean(obs, 3)[finite]
#     obs_filterd_var = var(obs, 3; corrected=false)[finite]
#     x = similar(x0, xdim, steps)
#     @views copy!(x[:,1], x0)
#     dx = similar(dx0, xdim, steps)
#     λ = zeros(T, θdim, steps)
#     dλ = zeros(T, θdim, steps)
#     Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(obs), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs, obs_filterd_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ)
# end
#
# function Adjoint(dt::T, obs::AbstractArray{T,3}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, x0::AbstractVector{T}, p::AbstractVector{T}) where {T<:AbstractFloat}
#     xdim, steps, replicates = size(obs)
#     θdim = xdim + length(p)
#     t = collect(0.:dt:dt*(steps-1))
#     obs_variance = similar(x0)
#     Nobs = [pseudo_Nobs[_i] + count(isfinite.(obs[_i,:,:])) for _i in 1:xdim]
#     finite = Matrix{Bool}(xdim, steps)
#     all!(finite, isfinite.(obs))
#     obs_filterd_mean = mean(obs, 3)[finite]
#     obs_filterd_var = var(obs, 3; corrected=false)[finite]
#     x = similar(x0, xdim, steps)
#     @views copy!(x[:,1], x0)
#     dx = similar(x0, xdim, steps)
#     dp = similar(p)
#     λ = zeros(T, θdim, steps)
#     dλ = zeros(T, θdim, steps)
#     Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(obs), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs, obs_filterd_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ)
# end
#
# function Adjoint(dt::T, total_T::T, obs_variance::AbstractVector{T}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, x0::AbstractVector{T}, p::AbstractVector{T}) where {T<:AbstractFloat}
#     steps = Int(total_T/dt) + 1
#     xdim = length(x0)
#     θdim = xdim + length(p)
#     t = collect(0.:dt:dt*(steps-1))
#     obs = similar(x0, xdim, steps, 1)
#     Nobs = copy(pseudo_Nobs)
#     finite = Matrix{Bool}(xdim, steps)
#     all!(finite, isfinite.(obs))
#     obs_filterd_mean = mean(obs, 3)[finite]
#     obs_filterd_var = var(obs, 3; corrected=false)[finite]
#     x = similar(x0, xdim, steps)
#     @views copy!(x[:,1], x0)
#     dx = similar(x0, xdim, steps)
#     dp = similar(p)
#     λ = zeros(T, θdim, steps)
#     dλ = zeros(T, θdim, steps)
#     Adjoint{xdim, θdim, 1, T, typeof(p), typeof(x), typeof(obs), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs, obs_filterd_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ)
# end

function Adjoint(dt::T, total_T::T, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_var::AbstractVector{T}, x0::AbstractVector{T}, p::AbstractVector{T}, replicates::Int) where {T<:AbstractFloat}
    steps = Int(total_T/dt) + 1
    xdim = length(x0)
    θdim = xdim + length(p)
    t = collect(0.:dt:dt*(steps-1))
    obs_variance = Vector{T}(xdim)
    Nobs = copy(pseudo_Nobs)
    pseudo_obs_TSS = pseudo_Nobs .* pseudo_obs_var
    finite = Matrix{Bool}(xdim, steps)
    obs_mean = Matrix{T}(xdim, steps)
    obs_filterd_var = Vector{T}(xdim)
    x = similar(x0, xdim, steps)
    @views copy!(x[:,1], x0)
    dx = similar(x0, xdim, steps)
    dp = similar(p)
    λ = zeros(T, θdim, steps)
    dλ = zeros(T, θdim, steps)
    Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ) # fixed bug. K=replicates.
end

function Adjoint(Nparams::Int, dt::T, obs::AbstractArray{T,3}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_var::AbstractVector{T}, x0::AbstractVector{T}) where {T<:AbstractFloat}
    xdim, steps, replicates = size(obs)
    θdim = xdim + Nparams
    t = collect(0.:dt:dt*(steps-1))
    obs_variance = Vector{T}(xdim)
    Nobs = copy(pseudo_Nobs)
    pseudo_obs_TSS = pseudo_Nobs .* pseudo_obs_var
    finite = Matrix{Bool}(xdim, steps)
    obs_filterd_var = Vector{T}(xdim)
    x = similar(x0, xdim, steps)
    @views copy!(x[:,1], x0)
    p = Vector{T}(Nparams)
    dx = similar(x0, xdim, steps)
    dp = similar(p)
    λ = zeros(T, θdim, steps)
    dλ = zeros(T, θdim, steps)

    all!(finite, isfinite.(obs))
    obs_mean = reshape(mean(obs, 3), xdim, steps)
    for _j in 1:xdim
        obs_filterd_var[_j] = mapreduce(x -> isnan(x) ? zero(x) : x, +, var(obs[_j,:,:], 2; corrected=false))
        Nobs[_j] .+= count(isfinite.(obs[_j,:,:]))
    end
    Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ) # fixed bug. K=replicates.
end


# function Adjoint(dt::T, obs::AbstractArray{T,3}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, p::AbstractVector{T}) where {T<:AbstractFloat}
#     xdim, steps, replicates = size(obs)
#     θdim = xdim + length(p)
#     t = collect(0.:dt:dt*(steps-1))
#     obs_variance = similar(obs, xdim)
#     Nobs = [pseudo_Nobs[_i] + count(isfinite.(obs[_i,:,:])) for _i in 1:xdim]
#     finite = Matrix{Bool}(xdim, steps)
#     all!(finite, isfinite.(obs))
#     obs_filterd_mean = mean(obs, 3)[finite]
#     obs_filterd_var = var(obs, 3; corrected=false)[finite]
#     x = similar(obs, xdim, steps)
#     dx = similar(obs, xdim, steps)
#     dp = similar(p)
#     λ = zeros(T, θdim, steps)
#     dλ = zeros(T, θdim, steps)
#     Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(obs), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs, obs_filterd_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ)
# end
#
# function Adjoint(dt::T, obs::AbstractArray{T,3}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, M::Int) where {T<:AbstractFloat}
#     xdim, steps, replicates = size(obs)
#     θdim = xdim + M
#     t = collect(0.:dt:dt*(steps-1))
#     obs_variance = similar(obs, xdim)
#     Nobs = [pseudo_Nobs[_i] + count(isfinite.(obs[_i,:,:])) for _i in 1:xdim]
#     finite = Matrix{Bool}(xdim, steps)
#     all!(finite, isfinite.(obs))
#     obs_filterd_mean = mean(obs, 3)[finite]
#     obs_filterd_var = var(obs, 3; corrected=false)[finite]
#     x = similar(obs, xdim, steps)
#     p = similar(obs, M)
#     dx = similar(x)
#     dp = similar(p)
#     λ = zeros(T, θdim, steps)
#     dλ = zeros(T, θdim, steps)
#     Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(obs), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs, obs_filterd_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ)
# end


mutable struct AssimilationResults{M, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix}
    p::Nullable{A}
    obs_variance::Nullable{A}
    stddev::Nullable{A}
    covariance::Nullable{B}
    AssimilationResults{M, T, A, B}(p::Nullable{S}, obs_variance::Nullable{S}, stddev::Nullable{S}, covariance::Nullable{U}) where {S<:AbstractVector{T}, U<:AbstractMatrix{T}} where {M,T,A,B} = new{M,T,A,B}(p, obs_variance, stddev, covariance)
end

function AssimilationResults(p::AbstractVector{T}, obs_variance::AbstractVector{T}, stddev::AbstractVector{T}, covariance::AbstractMatrix{T}) where {T<:AbstractFloat}
    M = length(p)
    AssimilationResults{M, T, typeof(p), typeof(covariance)}(Nullable(p), Nullable(obs_variance), Nullable(stddev), Nullable(covariance))
end

function AssimilationResults(p::AbstractVector{T}, obs_variance::AbstractVector{T}, covariance::AbstractMatrix{T}) where {T<:AbstractFloat}
    M = length(p)
    AssimilationResults{M, T, typeof(p), typeof(covariance)}(Nullable(p), Nullable(obs_variance), Nullable{Vector{T}}(), Nullable(covariance))
end

function AssimilationResults(p::AbstractVector{T}, obs_variance::AbstractVector{T}) where {T<:AbstractFloat}
    M = length(p)
    AssimilationResults{M, T, typeof(p), Matrix{T}}(Nullable(p), Nullable(obs_variance), Nullable{Vector{T}}(), Nullable{Matrix{T}}())
end

function AssimilationResults(M::Int, t::Type{T}) where {T<:AbstractFloat}
    AssimilationResults{M, t, Vector{t}, Matrix{t}}(Nullable{Vector{t}}(), Nullable{Vector{t}}(), Nullable{Vector{t}}(), Nullable{Matrix{t}}())
end
