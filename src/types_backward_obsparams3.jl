mutable struct Model{N, L, R, U, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix, C<:AbstractArray, F<:Function, G1<:Function, G2<:Function, H1<:Function, H2<:Function, H3<:Function, P<:Function, Q<:Function, QQ<:Function, I<:Function, J<:Function, K<:Function, KK<:Function, M<:Function} #<: AbstractModel{N, L, T, A, B, C, F, G, H}
    dxdt::A
    jacobianx::B
    jacobianp::B
    hessianxx::C
    hessianxp::C
    hessianpp::C
    observation::A
    observation_jacobianx::B
    observation_jacobianr::B
    observation_jacobianp::B
    observation_hessianxx::C
    observation_hessianxr::C
    observation_hessianrr::C
    observation_hessianpp::C
    dxdt!::F
    jacobianx!::G1
    jacobianp!::G2
    hessianxx!::H1
    hessianxp!::H2
    hessianpp!::H3
    observation!::I
    observation_jacobianx!::J
    observation_jacobianr!::K
    observation_jacobianp!::KK
    observation_hessianxx!::M
    observation_hessianxr!::P
    observation_hessianrr!::Q
    observation_hessianpp!::QQ
    time_point::A
    Δtime_point::A
    Model{N, L, R, U, T, A, B, C, F, G1, G2, H1, H2, H3, P, Q, QQ, I, J, K, KK, M}(dxdt::AbstractVector{T}, jacobianx::AbstractMatrix{T}, jacobianp::AbstractMatrix{T}, hessianxx::AbstractArray{T,3}, hessianxp::AbstractArray{T,3}, hessianpp::AbstractArray{T,3}, observation::AbstractVector{T}, observation_jacobianx::AbstractMatrix{T}, observation_jacobianr::AbstractMatrix{T}, observation_jacobianp::AbstractMatrix{T}, observation_hessianxx::AbstractArray{T,3}, observation_hessianxr::AbstractArray{T,3}, observation_hessianrr::AbstractArray{T,3}, observation_hessianpp::AbstractArray{T,3}, dxdt!::F, jacobianx!::G1, jacobianp!::G2, hessianxx!::H1, hessianxp!::H2, hessianpp!::H3, observation!::I, observation_jacobianx!::J, observation_jacobianr!::K, observation_jacobianp!::KK, observation_hessianxx!::M, observation_hessianxr!::P, observation_hessianrr!::Q, observation_hessianpp!::QQ, time_point::AbstractVector{T}, Δtime_point::AbstractVector{T}) where {N,L,R,U,T,A,B,C,F,G1,G2,H1,H2,H3,P,Q,QQ,I,J,K,KK,M} = new{N,L,R,U,T,A,B,C,F,G1,G2,H1,H2,H3,P,Q,QQ,I,J,K,KK,M}(dxdt, jacobianx, jacobianp, hessianxx, hessianxp, hessianpp, observation, observation_jacobianx, observation_jacobianr, observation_jacobianp, observation_hessianxx, observation_hessianxr, observation_hessianrr, observation_hessianpp, dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, observation!, observation_jacobianx!, observation_jacobianr!, observation_jacobianp!, observation_hessianxx!, observation_hessianxr!, observation_hessianrr!, observation_hessianpp!, time_point, Δtime_point)
end

function Model(t::Type{T}, N, L, R, U, time_point::AbstractVector{T}, dxdt!::F, jacobianx!::G1, jacobianp!::G2, hessianxx!::H1, hessianxp!::H2, hessianpp!::H3, observation!::I, observation_jacobianx!::J, observation_jacobianr!::K, observation_jacobianp!::KK, observation_hessianxx!::M, observation_hessianxr!::P, observation_hessianrr!::Q, observation_hessianpp!::QQ) where {T<:AbstractFloat, F<:Function, G1<:Function, G2<:Function, H1<:Function, H2<:Function, H3<:Function, P<:Function, Q<:Function, QQ<:Function, I<:Function, J<:Function, K<:Function, KK<:Function, M<:Function}
    _M = L-N
    dxdt = Array{t}(undef, N)
    jacobianx = zeros(t, N, N)
    jacobianp = zeros(t, N, _M)
    hessianxx = zeros(t, N, N, N)
    hessianxp = zeros(t, N, N, _M)
    hessianpp = zeros(t, N, _M, _M)
    observation = Array{t}(undef, U)
    observation_jacobianx = zeros(t, U, N)
    observation_jacobianr = zeros(t, U, R)
    observation_jacobianp = zeros(t, U, _M)
    observation_hessianxx = zeros(t, U, N, N)
    observation_hessianxr = zeros(t, U, N, R)
    observation_hessianrr = zeros(t, U, R, R)
    observation_hessianpp = zeros(t, U, _M, _M)

    Ts = length(time_point)

    Δtime_point = similar(time_point, Ts-1)
    for _i in 1:Ts-1
        Δtime_point[_i] = time_point[_i+1] - time_point[_i]
    end

    Model{N, L, R, U, t, typeof(dxdt), typeof(jacobianx), typeof(hessianxx), typeof(dxdt!), typeof(jacobianx!), typeof(jacobianp!), typeof(hessianxx!), typeof(hessianxp!), typeof(hessianpp!), typeof(observation_hessianxr!), typeof(observation_hessianrr!), typeof(observation_hessianpp!), typeof(observation!), typeof(observation_jacobianx!), typeof(observation_jacobianr!), typeof(observation_jacobianp!), typeof(observation_hessianxx!)}(dxdt, jacobianx, jacobianp, hessianxx, hessianxp, hessianpp, observation, observation_jacobianx, observation_jacobianr, observation_jacobianp, observation_hessianxx, observation_hessianxr, observation_hessianrr, observation_hessianpp, dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, observation!, observation_jacobianx!, observation_jacobianr!, observation_jacobianp!, observation_hessianxx!, observation_hessianxr!, observation_hessianrr!, observation_hessianpp!, time_point, Δtime_point)
end

mutable struct Adjoint{N, L, R, U, K, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix, D<:AbstractVector, E<:AbstractMatrix}
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
    # r::A
    dx::B
    dp::A
    λ::B
    dλ::B
    res::A
    jacobian_inv::B
    trylimit::Int
    newton_tol::T
    regularization::T
    Adjoint{N, L, R, U, K, T, A, B, D, E}(dt::T, steps::Int, t::AbstractVector{T}, obs_variance::AbstractVector{T}, obs_mean::AbstractMatrix{T}, obs_filterd_var::AbstractVector{T}, finite::AbstractMatrix{Bool}, Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}, res::AbstractVector{T}, jacobian_inv::AbstractMatrix{T}, trylimit::Int, newton_tol::T, regularization::T) where {N,L,R,U,K,T,A,B,D,E} = new{N,L,R,U,K,T,A,B,D,E}(dt, steps, t, obs_variance, obs_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ, res, jacobian_inv, trylimit, newton_tol, regularization)
end

function Adjoint(dt::T, total_T::T, Nobsparams::Int, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_var::AbstractVector{T}, x0::AbstractVector{T}, p::AbstractVector{T}, replicates::Int, trylimit::Int, newton_tol::T, regularization::T) where {T<:AbstractFloat}
    steps = Int(total_T/dt) + 1
    xdim = length(x0)
    θdim = xdim + length(p) - Nobsparams
    rdim = Nobsparams
    obsdim = length(pseudo_obs_var)
    t = collect(0.:dt:dt*(steps-1))
    obs_variance = Vector{T}(undef, obsdim)
    Nobs = copy(pseudo_Nobs)
    pseudo_obs_TSS = pseudo_Nobs .* pseudo_obs_var
    finite = Matrix{Bool}(undef, obsdim, steps)
    obs_mean = Matrix{T}(undef, obsdim, steps)
    obs_filterd_var = Vector{T}(undef, obsdim)
    x = similar(x0, xdim, steps)
    @views copyto!(x[:,1], x0)
    dx = similar(x0, xdim, steps)
    dp = similar(p)
    λ = zeros(T, θdim + rdim, steps)
    dλ = zeros(T, θdim + rdim, steps)
    res = zeros(T, xdim)
    jacobian = similar(x0, xdim, xdim)
    Adjoint{xdim, θdim, rdim, obsdim, replicates, T, typeof(p), typeof(x), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ, res, jacobian, trylimit, newton_tol, regularization) # fixed bug. K=replicates.
end

function Adjoint(xdim::Int, Nparams::Int, Nobsparams::Int, dt::T, obs::AbstractArray{T,3}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_var::AbstractVector{T}, x0::AbstractVector{T}, trylimit::Int, newton_tol::T, regularization::T) where {T<:AbstractFloat}
    obsdim, steps, replicates = size(obs)
    θdim = xdim + Nparams
    t = collect(0.:dt:dt*(steps-1))
    obs_variance = Vector{T}(undef, obsdim)
    Nobs = copy(pseudo_Nobs)
    pseudo_obs_TSS = pseudo_Nobs .* pseudo_obs_var
    finite = Matrix{Bool}(undef, obsdim, steps)
    obs_filterd_var = Vector{T}(undef, obsdim)
    x = similar(x0, xdim, steps)
    @views copyto!(x[:,1], x0)
    p = Vector{T}(undef, Nparams + Nobsparams)
    dx = similar(x0, xdim, steps)
    dp = similar(p)
    λ = zeros(T, θdim + Nobsparams, steps)
    dλ = zeros(T, θdim + Nobsparams, steps)
    res = zeros(T, xdim)
    jacobian = similar(x0, xdim, xdim)

    all!(finite, isfinite.(obs))
    obs_mean = reshape(mean(obs, 3), obsdim, steps)
    for _j in 1:obsdim
        obs_filterd_var[_j] = mapreduce(x -> isnan(x) ? zero(x) : x, +, var(obs[_j,:,:], 2; corrected=false))
        Nobs[_j] .+= count(isfinite.(obs[_j,:,:]))
    end
    Adjoint{xdim, θdim, Nobsparams, obsdim, replicates, T, typeof(p), typeof(x), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, r, dx, dp, λ, dλ, res, jacobian, trylimit, newton_tol, regularization) # fixed bug. K=replicates.
end

mutable struct AssimilationResults{L, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix}
    θ::Union{A,Nothing}
    obs_variance::Union{A,Nothing}
    stddev::Union{A,Nothing}
    precision::Union{B,Nothing}
    covariance::Union{B,Nothing}
    AssimilationResults{L, T, A, B}(θ::Union{S,Nothing}, obs_variance::Union{S,Nothing}, stddev::Union{S,Nothing}, precision::Union{U,Nothing}, covariance::Union{U,Nothing}) where {S<:AbstractVector{T}, U<:AbstractMatrix{T}} where {L,T,A,B} = new{L,T,A,B}(θ, obs_variance, stddev, precision, covariance)
end

function AssimilationResults(θ::AbstractVector{T}, obs_variance::AbstractVector{T}, stddev::AbstractVector{T}, precision::AbstractMatrix{T}, covariance::AbstractMatrix{T}) where {T<:AbstractFloat}
    L = length(θ)
    AssimilationResults{L, T, typeof(θ), typeof(precision)}(θ, obs_variance, stddev, precision, covariance)
end

function AssimilationResults(θ::AbstractVector{T}, obs_variance::AbstractVector{T}, precision::AbstractMatrix{T}, covariance::AbstractMatrix{T}) where {T<:AbstractFloat}
    L = length(θ)
    AssimilationResults{L, T, typeof(θ), typeof(precision)}(θ, obs_variance, nothing, precision, covariance)
end

function AssimilationResults(θ::AbstractVector{T}, obs_variance::AbstractVector{T}, precision::AbstractMatrix{T}) where {T<:AbstractFloat}
    L = length(θ)
    AssimilationResults{L, T, typeof(θ), typeof(precision)}(θ, obs_variance, nothing, precision, nothing)
end

function AssimilationResults(θ::AbstractVector{T}, obs_variance::AbstractVector{T}) where {T<:AbstractFloat}
    L = length(θ)
    AssimilationResults{L, T, typeof(θ), Matrix{T}}(θ, obs_variance, nothing, nothing, nothing)
end

function AssimilationResults(L::Int, t::Type{T}) where {T<:AbstractFloat}
    AssimilationResults{L, t, Vector{t}, Matrix{t}}(nothing, nothing, nothing, nothing, nothing)
end
