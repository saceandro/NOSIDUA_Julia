mutable struct Model{N, L, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix, C<:AbstractArray, F<:Function, G1<:Function, G2<:Function, H1<:Function, H2<:Function, H3<:Function, O<:Function, I<:Function, J<:Function, K<:Function, M<:Function, S<:LinAlg.LU} #<: AbstractModel{N, L, T, A, B, C, F, G, H}
    dxdt::A
    jacobianx::B
    jacobianp::B
    hessianxx::C
    hessianxp::C
    hessianpp::C
    dxdt!::F
    jacobianx!::G1
    jacobianp!::G2
    hessianxx!::H1
    hessianxp!::H2
    hessianpp!::H3
    calc_eqdot!::O
    observation::I
    d_observation::J
    dd_observation::K
    inv_observation::M
    time_point::A
    Δtime_point::A
    eqdot::A
    rhs::A
    spline_lhs::S
    spline_rhs::B
    spline_lhs_inv::B
    spline_lhs_inv_rhs::B
    Model{N, L, T, A, B, C, F, G1, G2, H1, H2, H3, O, I, J, K, M, S}(dxdt::AbstractVector{T}, jacobianx::AbstractMatrix{T}, jacobianp::AbstractMatrix{T}, hessianxx::AbstractArray{T,3}, hessianxp::AbstractArray{T,3}, hessianpp::AbstractArray{T,3}, dxdt!::F, jacobianx!::G1, jacobianp!::G2, hessianxx!::H1, hessianxp!::H2, hessianpp!::H3, calc_eqdot!::O, observation::I, d_observation::J, dd_observation::K, inv_observation::M, time_point::AbstractVector{T}, Δtime_point::AbstractVector{T}, eqdot::AbstractVector{T}, rhs::AbstractVector{T}, spline_lhs::LinAlg.LU{T,Tridiagonal{T}}, spline_rhs::AbstractMatrix{T}, spline_lhs_inv::AbstractMatrix{T}, spline_lhs_inv_rhs::AbstractMatrix{T}) where {N,L,T,A,B,C,F,G1,G2,H1,H2,H3,O,I,J,K,M,S} = new{N,L,T,A,B,C,F,G1,G2,H1,H2,H3,O,I,J,K,M,S}(dxdt, jacobianx, jacobianp, hessianxx, hessianxp, hessianpp, dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, calc_eqdot!, observation, d_observation, dd_observation, inv_observation, time_point, Δtime_point, eqdot, rhs, spline_lhs, spline_rhs, spline_lhs_inv, spline_lhs_inv_rhs)
end

function Model(t::Type{T}, N, L, time_point::AbstractVector{T}, dxdt!::F, jacobianx!::G1, jacobianp!::G2, hessianxx!::H1, hessianxp!::H2, hessianpp!::H3, calc_eqdot!::O, observation::I, d_observation::J, dd_observation::K, inv_observation::M) where {T<:AbstractFloat, F<:Function, G1<:Function, G2<:Function, H1<:Function, H2<:Function, H3<:Function, O<:Function, I<:Function, J<:Function, K<:Function, M<:Function}
    _M = L-N
    dxdt = Array{t}(N)
    jacobianx = zeros(t, N, N)
    jacobianp = zeros(t, N, _M)
    hessianxx = zeros(t, N, N, N)
    hessianxp = zeros(t, N, N, _M)
    hessianpp = zeros(t, N, _M, _M)
    Ts = length(time_point)

    Δtime_point = similar(time_point, Ts-1)
    for _i in 1:Ts-1
        Δtime_point[_i] = time_point[_i+1] - time_point[_i]
    end

    eqdot = Array{t}(Ts)

    d = similar(time_point, Ts)
    dl = similar(time_point, Ts-1)
    du = similar(time_point, Ts-1)

    d[1] = 1.
    for _i in 2:Ts-1
        d[_i] = inv(Δtime_point[_i-1]) + inv(Δtime_point[_i])
    end
    d[Ts] = 1.
    d .*= 2.

    for _i in 1:Ts-2
        dl[_i] = inv(Δtime_point[_i])
    end
    dl[Ts-1] = 1.

    du[1] = 1.
    for _i in 2:Ts-1
        du[_i] = inv(Δtime_point[_i])
    end

    spline_lhs = factorize(Tridiagonal(dl, d, du))
    spline_lhs_inv = inv(Tridiagonal(dl, d, du))

    d[1] = -inv(Δtime_point[1])
    for _i in 2:Ts-1
        d[_i] = Δtime_point[_i-1]^(-2) - Δtime_point[_i]^(-2)
    end
    d[Ts] = inv(Δtime_point[end])

    for _i in 1:Ts-2
        dl[_i] = -(Δtime_point[_i]^(-2))
    end
    dl[Ts-1] = -inv(Δtime_point[Ts-1])

    du[1] = inv(Δtime_point[1])
    for _i in 2:Ts-1
        du[_i] = Δtime_point[_i]^(-2)
    end

    spline_rhs = Tridiagonal(3 .* dl, 3 .* d, 3 .* du)

    spline_lhs_inv_rhs = spline_lhs \ spline_rhs

    rhs = Array{t}(Ts)

    Model{N, L, t, typeof(dxdt), typeof(jacobianx), typeof(hessianxx), typeof(dxdt!), typeof(jacobianx!), typeof(jacobianp!), typeof(hessianxx!), typeof(hessianxp!), typeof(hessianpp!), typeof(calc_eqdot!), typeof(observation), typeof(d_observation), typeof(dd_observation), typeof(inv_observation), typeof(spline_lhs)}(dxdt, jacobianx, jacobianp, hessianxx, hessianxp, hessianpp, dxdt!, jacobianx!, jacobianp!, hessianxx!, hessianxp!, hessianpp!, calc_eqdot!, observation, d_observation, dd_observation, inv_observation, time_point, Δtime_point, eqdot, rhs, spline_lhs, spline_rhs, spline_lhs_inv, spline_lhs_inv_rhs)
end

# mutable struct Adjoint{N, L, K, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix, D<:AbstractVector, E<:AbstractMatrix, F<:AbstractVector}
#     dt::T
#     steps::Int
#     t::A
#     obs_variance::A
#     obs_mean::B
#     obs_filterd_var::A
#     finite::E
#     Nobs::D
#     pseudo_obs_TSS::A
#     x::B # dim(x) = N * (steps+1)
#     p::A
#     dx::B
#     dp::A
#     λ::B
#     dλ::B
#     jacobian_inv::B
#     trylimit::Int
#     search_box::F
#     Adjoint{N, L, K, T, A, B, D, E,F}(dt::T, steps::Int, t::AbstractVector{T}, obs_variance::AbstractVector{T}, obs_mean::AbstractMatrix{T}, obs_filterd_var::AbstractVector{T}, finite::AbstractMatrix{Bool}, Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}, jacobian_inv::AbstractMatrix{T}, trylimit::Int, search_box::AbstractVector{Uniform{T}}) where {N,L,K,T,A,B,D,E,F} = new{N,L,K,T,A,B,D,E,F}(dt, steps, t, obs_variance, obs_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ, jacobian_inv, trylimit, search_box)
# end

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
    res::A
    jacobian_inv::B
    trylimit::Int
    newton_tol::T
    regularization::T
    Adjoint{N, L, K, T, A, B, D, E}(dt::T, steps::Int, t::AbstractVector{T}, obs_variance::AbstractVector{T}, obs_mean::AbstractMatrix{T}, obs_filterd_var::AbstractVector{T}, finite::AbstractMatrix{Bool}, Nobs::AbstractVector{Int}, pseudo_obs_TSS::AbstractVector{T}, x::AbstractMatrix{T}, p::AbstractVector{T}, dx::AbstractMatrix{T}, dp::AbstractVector{T}, λ::AbstractMatrix{T}, dλ::AbstractMatrix{T}, res::AbstractVector{T}, jacobian_inv::AbstractMatrix{T}, trylimit::Int, newton_tol::T, regularization::T) where {N,L,K,T,A,B,D,E} = new{N,L,K,T,A,B,D,E}(dt, steps, t, obs_variance, obs_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ, res, jacobian_inv, trylimit, newton_tol, regularization)
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

# function Adjoint(dt::T, total_T::T, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_var::AbstractVector{T}, x0::AbstractVector{T}, p::AbstractVector{T}, replicates::Int, trylimit::Int, search_box_nlsolver::AbstractVector{T}) where {T<:AbstractFloat}
#     steps = Int(total_T/dt) + 1
#     xdim = length(x0)
#     θdim = xdim + length(p)
#     t = collect(0.:dt:dt*(steps-1))
#     obs_variance = Vector{T}(xdim)
#     Nobs = copy(pseudo_Nobs)
#     pseudo_obs_TSS = pseudo_Nobs .* pseudo_obs_var
#     finite = Matrix{Bool}(xdim, steps)
#     obs_mean = Matrix{T}(xdim, steps)
#     obs_filterd_var = Vector{T}(xdim)
#     x = similar(x0, xdim, steps)
#     @views copy!(x[:,1], x0)
#     dx = similar(x0, xdim, steps)
#     dp = similar(p)
#     λ = zeros(T, θdim, steps)
#     dλ = zeros(T, θdim, steps)
#     jacobian = similar(x0, xdim, xdim)
#     search_box = [Uniform(-search_box_nlsolver[i], search_box_nlsolver[i]) for i in 1:xdim]
#     Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(Nobs), typeof(finite), typeof(search_box)}(dt, steps-1, t, obs_variance, obs_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ, jacobian, trylimit, search_box) # fixed bug. K=replicates.
# end
#
# function Adjoint(Nparams::Int, dt::T, obs::AbstractArray{T,3}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_var::AbstractVector{T}, x0::AbstractVector{T}, trylimit::Int, search_box_nlsolver::AbstractVector{T}) where {T<:AbstractFloat}
#     xdim, steps, replicates = size(obs)
#     θdim = xdim + Nparams
#     t = collect(0.:dt:dt*(steps-1))
#     obs_variance = Vector{T}(xdim)
#     Nobs = copy(pseudo_Nobs)
#     pseudo_obs_TSS = pseudo_Nobs .* pseudo_obs_var
#     finite = Matrix{Bool}(xdim, steps)
#     obs_filterd_var = Vector{T}(xdim)
#     x = similar(x0, xdim, steps)
#     @views copy!(x[:,1], x0)
#     p = Vector{T}(Nparams)
#     dx = similar(x0, xdim, steps)
#     dp = similar(p)
#     λ = zeros(T, θdim, steps)
#     dλ = zeros(T, θdim, steps)
#     jacobian = similar(x0, xdim, xdim)
#     search_box = [Uniform(-search_box_nlsolver[i], search_box_nlsolver[i]) for i in 1:xdim]
#
#     all!(finite, isfinite.(obs))
#     obs_mean = reshape(mean(obs, 3), xdim, steps)
#     for _j in 1:xdim
#         obs_filterd_var[_j] = mapreduce(x -> isnan(x) ? zero(x) : x, +, var(obs[_j,:,:], 2; corrected=false))
#         Nobs[_j] .+= count(isfinite.(obs[_j,:,:]))
#     end
#     Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(Nobs), typeof(finite), typeof(search_box)}(dt, steps-1, t, obs_variance, obs_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ, jacobian, trylimit, search_box) # fixed bug. K=replicates.
# end

# function Adjoint(xdim::Int, dt::T, total_T::T, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_var::AbstractVector{T}, p::AbstractVector{T}, replicates::Int, trylimit::Int, newton_tol::T, regularization::T) where {T<:AbstractFloat}
#     steps = Int(total_T/dt) + 1
#     θdim = xdim + length(p)
#     t = collect(0.:dt:dt*(steps-1))
#     obs_variance = Vector{T}(xdim)
#     Nobs = copy(pseudo_Nobs)
#     pseudo_obs_TSS = pseudo_Nobs .* pseudo_obs_var
#     finite = Matrix{Bool}(xdim, steps)
#     obs_mean = Matrix{T}(xdim, steps)
#     obs_filterd_var = Vector{T}(xdim)
#     x = similar(obs_variance, xdim, steps)
#     dx = similar(x0, xdim, steps)
#     dp = similar(p)
#     λ = zeros(T, θdim, steps)
#     dλ = zeros(T, θdim, steps)
#     res = zeros(T, xdim)
#     jacobian = similar(x0, xdim, xdim)
#     Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ, res, jacobian, trylimit, newton_tol, regularization) # fixed bug. K=replicates.
# end

function Adjoint(dt::T, total_T::T, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_var::AbstractVector{T}, x0::AbstractVector{T}, p::AbstractVector{T}, replicates::Int, trylimit::Int, newton_tol::T, regularization::T) where {T<:AbstractFloat}
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
    res = zeros(T, xdim)
    jacobian = similar(x0, xdim, xdim)
    Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ, res, jacobian, trylimit, newton_tol, regularization) # fixed bug. K=replicates.
end

function Adjoint(Nparams::Int, dt::T, obs::AbstractArray{T,3}, pseudo_Nobs::AbstractVector{Int}, pseudo_obs_var::AbstractVector{T}, x0::AbstractVector{T}, trylimit::Int, newton_tol::T, regularization::T) where {T<:AbstractFloat}
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
    res = zeros(T, xdim)
    jacobian = similar(x0, xdim, xdim)

    all!(finite, isfinite.(obs))
    obs_mean = reshape(mean(obs, 3), xdim, steps)
    for _j in 1:xdim
        obs_filterd_var[_j] = mapreduce(x -> isnan(x) ? zero(x) : x, +, var(obs[_j,:,:], 2; corrected=false))
        Nobs[_j] .+= count(isfinite.(obs[_j,:,:]))
    end
    Adjoint{xdim, θdim, replicates, T, typeof(p), typeof(x), typeof(Nobs), typeof(finite)}(dt, steps-1, t, obs_variance, obs_mean, obs_filterd_var, finite, Nobs, pseudo_obs_TSS, x, p, dx, dp, λ, dλ, res, jacobian, trylimit, newton_tol, regularization) # fixed bug. K=replicates.
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


mutable struct AssimilationResults{L, T<:AbstractFloat, A<:AbstractVector, B<:AbstractMatrix}
    θ::Nullable{A}
    obs_variance::Nullable{A}
    stddev::Nullable{A}
    precision::Nullable{B}
    covariance::Nullable{B}
    AssimilationResults{L, T, A, B}(θ::Nullable{S}, obs_variance::Nullable{S}, stddev::Nullable{S}, precision::Nullable{U}, covariance::Nullable{U}) where {S<:AbstractVector{T}, U<:AbstractMatrix{T}} where {L,T,A,B} = new{L,T,A,B}(θ, obs_variance, stddev, precision, covariance)
end

function AssimilationResults(θ::AbstractVector{T}, obs_variance::AbstractVector{T}, stddev::AbstractVector{T}, precision::AbstractMatrix{T}, covariance::AbstractMatrix{T}) where {T<:AbstractFloat}
    L = length(θ)
    AssimilationResults{L, T, typeof(θ), typeof(precision)}(Nullable(θ), Nullable(obs_variance), Nullable(stddev), Nullable(precision), Nullable(covariance))
end

function AssimilationResults(θ::AbstractVector{T}, obs_variance::AbstractVector{T}, precision::AbstractMatrix{T}, covariance::AbstractMatrix{T}) where {T<:AbstractFloat}
    L = length(θ)
    AssimilationResults{L, T, typeof(θ), typeof(precision)}(Nullable(θ), Nullable(obs_variance), Nullable{Vector{T}}(), Nullable(precision), Nullable(covariance))
end

function AssimilationResults(θ::AbstractVector{T}, obs_variance::AbstractVector{T}, precision::AbstractMatrix{T}) where {T<:AbstractFloat}
    L = length(θ)
    AssimilationResults{L, T, typeof(θ), typeof(precision)}(Nullable(θ), Nullable(obs_variance), Nullable{Vector{T}}(), Nullable(precision), Nullable{Matrix{T}}())
end

function AssimilationResults(θ::AbstractVector{T}, obs_variance::AbstractVector{T}) where {T<:AbstractFloat}
    L = length(θ)
    AssimilationResults{L, T, typeof(θ), Matrix{T}}(Nullable(θ), Nullable(obs_variance), Nullable{Vector{T}}(), Nullable{Matrix{T}}(), Nullable{Matrix{T}}())
end

function AssimilationResults(L::Int, t::Type{T}) where {T<:AbstractFloat}
    AssimilationResults{L, t, Vector{t}, Matrix{t}}(Nullable{Vector{t}}(), Nullable{Vector{t}}(), Nullable{Vector{t}}(), Nullable{Matrix{t}}(), Nullable{Matrix{t}}())
end
