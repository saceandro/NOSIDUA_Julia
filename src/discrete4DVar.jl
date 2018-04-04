module Discrete4DVar

using CatViews, DataArrays, NLSolversBase, Optim

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
    jacobian = zeros(T, xdim, θdim)
    hessian = zeros(T, xdim, θdim, θdim)
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

function dxdt!(a::Adjoint{N}, t, x) where {N}
    a.dxdt[1]     = a.p[2] * (x[2]   - x[N-1]) * x[N]   + a.p[1] - x[1]
    a.dxdt[2]     = a.p[2] * (x[3]   - x[N])   * x[1]   + a.p[1] - x[2]
    @simd for i in 3:N-1
        a.dxdt[i] = a.p[2] * (x[i+1] - x[i-2]) * x[i-1] + a.p[1] - x[i]
    end
    a.dxdt[N]     = a.p[2] * (x[1]   - x[N-2]) * x[N-1] + a.p[1] - x[N]
    nothing
end

# function jacobian!(a::Adjoint{N, L}, t, x) where {N, L} # might be faster if SparseMatrixCSC is used. L=N+M.
#     println((N+1 == 6) + 0.)
#     for j in 1:L, i in 1:N
#         a.jacobian[i,j] = a.p[2]     * ( (mod1(i+1, N) == j) -  (mod1(i-2, N) == j)) * x[mod1(i-1, N)]
#                         + a.p[2]     * (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      *  (mod1(i-1, N) == j)
#                         + (N+2 == j) * (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      * x[mod1(i-1, N)]
#                         + (N+1 == j)
#                         - (i   == j)
#     println(a.jacobian)
#     end
#     nothing
# end

function jacobian!(a::Adjoint{N}, t, x) where {N} # might be faster if SparseMatrixCSC is used. L=N+M. need to write in one line
    for j in 1:N, i in 1:N
        a.jacobian[i,j] = a.p[2]     * ( (mod1(i+1, N) == j) -  (mod1(i-2, N) == j)) * x[mod1(i-1, N)] + a.p[2]     * (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      *  (mod1(i-1, N) == j) - (i   == j)
    end
    for i in 1:N
        a.jacobian[i,N+1] = 1.
    end
    for i in 1:N
        a.jacobian[i,N+2] = (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      * x[mod1(i-1, N)]
    end
    nothing
end

# function hessian!(a::Adjoint{N, L}, t, x) where {N, L}
#     for k in 1:L, j in 1:L, i in 1:N
#         a.hessian[i,j,k] = (N+2 == j) * ( (mod1(i+1, N)==k) -  (mod1(i-2, N)==k)) * x[mod1(i-1, N)]
#                          + (N+2 == j) * (x[mod1(i+1, N)]    - x[mod1(i-2, N)])    *  (mod1(i-1, N)==k)
#                          + (N+2 == k) * ( (mod1(i+1, N)==j) -  (mod1(i-2, N)==j)) * x[mod1(i-1, N)]
#                          + a.p[2]     * ( (mod1(i+1, N)==j) -  (mod1(i-2, N)==j)) *  (mod1(i-1, N)==k)
#                          + (N+2 == k) * (x[mod1(i+1, N)]    - x[mod1(i-2, N)])    *  (mod1(i-1, N)==j)
#                          + a.p[2]     * ( (mod1(i+1, N)==k) -  (mod1(i-2, N)==k)) *  (mod1(i-1, N)==j)
#     end
#     nothing
# end

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

innovation_λ(  x, obs, obs_variance) = isna(obs) ? zero(x)  : (x - obs)/obs_variance # element-wise innovation of λ. fix me to be type invariant!
# innovation_λ(  x::T, obs::T, obs_variance::T) where {T <: AbstractFloat} = (x - obs)/obs_variance
# innovation_λ(  x::T, obs::NAtype, obs_variance::T) where {T <: AbstractFloat} = zero(x)

innovation_dλ(dx, obs, obs_variance) = isna(obs) ? zero(dx) : dx/obs_variance        # element-wise innovation of dλ.

next_x!(        a::Adjoint,       t, x, x_nxt)                                               = (dxdt!(    a, t, x);                                  x_nxt .=  x .+ a.dxdt                                                             .* a.dt; nothing)

next_dx!(       a::Adjoint,       t, x,        dx, dx_nxt)                                   = (jacobian!(a, t, x);                                 dx_nxt .= dx .+ a.jacobian  * CatView(dx, a.dp)                                    .* a.dt; nothing)

@views prev_λ!( a::Adjoint{N},    t, x,                   λ, λ_prev)            where {N}    = (jacobian!(a, t, x);                                 λ_prev .=  λ .+ a.jacobian' *  λ[1:N]                                              .* a.dt; nothing)

@views prev_dλ!(a::Adjoint{N, L}, t, x,        dx,        λ,       dλ, dλ_prev) where {N, L} = (hessian!( a, t, x); prev_λ!(a, t, x, dλ, dλ_prev); dλ_prev .+= reshape(reshape(a.hessian, N*L, L) * CatView(dx, a.dp), N, L)' * λ[1:N] .* a.dt; nothing)


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
    a.λ[N+1:end,end] .= 0.
    for _i in a.steps:-1:1
        prev_λ!(a, a.dt*(_i-1), a.x[:,_i],                                      a.λ[:,_i+1], a.λ[:,_i]) # fix me! a.dt*_i?
        a.λ[1:N,_i] .+= innovation_λ.(a.x[:,_i], a.obs[:,_i], a.obs_variance)
    end
    nothing
end

@views function hessian_vector_product!(a::Adjoint{N}) where {N} # assuming dx[:,1] .= dx0; dp .= dp; neighboring!(jacobian, dt, t, x, p, dx, dp); is already run. dλ[:,1] is the hessian_vector_product.
    a.dλ[1:N,end] .= innovation_dλ.(a.dx[:,end], a.obs[:,end], a.obs_variance)
    a.dλ[N+1:end,end] .= 0.
    for _i in a.steps:-1:1
        prev_dλ!(a, a.dt*(_i-1), a.x[:,_i],            a.dx[:,_i],              a.λ[:,_i+1],            a.dλ[:,_i+1], a.dλ[:,_i]) # fix me! a.dt*_i?
        a.dλ[1:N,_i] .+= innovation_dλ.(a.dx[:,_i], a.obs[:,_i], a.obs_variance)
    end
    nothing
end

cost(a) = mapreduce(abs2, +, a.x .- a.obs, skipna=true) / a.obs_variance / oftype(a.obs_variance, 2.) # assuming x[:,1] .= x0; orbit!(dxdt, t, x, p, dt); is already run

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

@views function fg!(F, ∇θ, θ, a::Adjoint{N}) where {N}
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

@views function minimize!(initial_θ, a::Adjoint{N,L}) where {N,L} # Fixed. views is definitely needed for copy!
    copy!(a.x[:,1], initial_θ[1:N])
    copy!(a.p, initial_θ[N+1:end])
    orbit!(a)
    F = cost(a)
    println("F: $F")

    gradient!(a)
    ∇θ = zeros(L)
    copy!(∇θ, a.λ[:,1])
    println("∇θ: $∇θ")

    df = OnceDifferentiable(θ -> fg!(F, nothing, θ, a), (∇θ, θ) -> fg!(nothing, ∇θ, θ, a), (∇θ, θ) -> fg!(F, ∇θ, θ, a), initial_θ, F, ∇θ, inplace=true)
    hoge = optimize(df, initial_θ, BFGS())
    println(hoge)
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
using CSV, DataFrames, Discrete4DVar
# obs = @data [-1.15567    0.983238   1.3993    -0.221607   0.399947   0.396658  -0.376941   0.569227  -0.466949  -1.01592     NA; 0.513092  -0.472495   1.56686   -1.04834   -0.478666   0.156926   0.497267  -0.150043  -0.174981  -1.28425    -1.18386; 0.820008   0.935917  -2.09416   -0.356491   0.312809  -0.872522  -0.226705  -1.20098   -0.784198   0.0891215  -1.03633; 0.624134   0.339597   0.528548   0.962726  -0.396218  -0.705182  -0.392724  -0.756412  -1.45077   -0.105439   -1.04203; 1.47352  -0.824375  -0.276906  -1.77185    0.475747  -1.5462  -1.70126    0.192254   0.553551   1.54488     0.35927]
# a = Adjoint(0.01, 100, 1., obs, randn(5), randn(2), randn(5), randn(2))
# initial_θ = randn(7)
# minimize!(initial_θ, a)
# hessian = similar(a.x, 7, 7)
# covariance = similar(a.x, 7, 7)
# variance = similar(a.x, 7)
# stddev = similar(a.x, 7)
# covariance!(hessian, covariance, variance, stddev, a)

const N = 5
const true_p = [8., 1.]
const obs_variance = 1.
const obs_iteration = 5
const dt = 0.01
const spinup = 73.
const T = 1.
const steps = Integer(T/dt)
const generation_seed = 0

const t = collect(0.:.01:1.)

const pref = "../../adjoint/data/Lorenz96/N_$N/p_$(join(true_p, "_"))/obsvar_$obs_variance/obsiter_$obs_iteration/dt_$dt/spinup_$spinup/T_$T/seed_$generation_seed/"

const tob = readdlm(pref * "true.tsv")'
const _obs = readdlm(pref * "observed.tsv")'
obs = data(_obs)
obs[isnan.(obs.data)] = NA
const tob_df = CSV.read(pref * "true.tsv", nullable=false, delim="\t", header=false)
const obs_df = CSV.read(pref * "observed.tsv", nullable=false, delim="\t", header=false)

using PlotlyJS

const mask = [!any(isnan.(_obs[1:3,_i])) for _i in 1:101]
const obs_pl = scatter3d(;x=_obs[1,:][mask], y=_obs[2,:][mask], z=_obs[3,:][mask], mode="lines", line=attr(color="red", width=5))
const true_pl = scatter3d(;x=tob[1,:],y=tob[2,:], z=tob[3,:], mode="lines", line=attr(color="#1f77b4", width=5))
PlotlyJS.plot([true_pl, obs_pl])


using Gadfly

const x0 = randn(5)
const p = randn(2)
const dx0 = [1., 0., 0., 0., 0.]
const dp = zeros(2)
a = Adjoint(dt, steps, obs_variance, obs, x0, p, dx0, dp)
initial_θ = randn(7)
initial_θ[6] = 7.5
initial_θ[7] = 0.8
minimize!(initial_θ, a)
hessian = similar(a.x, 7, 7)
covariance = similar(a.x, 7, 7)
variance = similar(a.x, 7)
stddev = similar(a.x, 7)
covariance!(hessian, covariance, variance, stddev, a)
const true_orbit = scatter3d(;x=tob[1,:],y=tob[2,:], z=tob[3,:], mode="lines", line=attr(color="#1f77b4", width=2))
const assimilated = scatter3d(;x=a.x[1,:], y=a.x[2,:], z=a.x[3,:], mode="lines", line=attr(color="yellow", width=2))
PlotlyJS.plot([true_orbit, assimilated])
PlotlyJS.plot([true_orbit, assimilated, obs_pl])

const white_panel = Theme(panel_fill="white")
p_stack = Array{Gadfly.Plot}(0)
for _i in 1:N
    df_tob = DataFrame(t=t, x=tob[_i,:], data_type="true orbit")
    _mask = .!obs.na[_i,:]
    df_obs = DataFrame(t=t[_mask], x=obs[_i,:][_mask], data_type="observed")
    # df_obs = DataFrame(x=t, y=obs[_i,:], label="observed")
    df_assim = DataFrame(t=t, x=a.x[_i,:], data_type="assimilated")
    df_all = vcat(df_tob, df_obs, df_assim)
    p_stack = vcat(p_stack, Gadfly.plot(df_all, x="t", y="x", color="data_type", Geom.line, Scale.color_discrete_manual("blue", "red", "green"), Guide.colorkey(title=""), Guide.ylabel("x<sub>$_i</sub>"), white_panel))
end
draw(PDF("lorenz.pdf", 24cm, 40cm), vstack(p_stack))
# Gadfly.plot(layer(x=t, y=tob[1,:], Geom.line, Theme(default_color="blue")), layer(x=t, y=a.x[1,:], Geom.line, Theme(default_color="green")), layer(x=t[!obs.na[1,:]], y=obs[1,:][!obs.na[1,:]], Geom.line, Theme(default_color="red")))
