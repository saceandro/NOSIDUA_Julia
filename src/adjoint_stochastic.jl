@views function innovation_λ!(innovation, m::Model{N,L,R,U}, t, x, p, obs_mean, obs_variance_over_K, finite) where {N,L,R,U}
    fill!(innovation, 0.)
    m.observation!(m, t, x, p)
    m.observation_jacobianx!(m, t, x, p)
    for _j in 1:U
        if finite[_j]
            innovation .+= (m.observation[_j] - obs_mean[_j]) / obs_variance_over_K[_j] .* m.observation_jacobianx[_j,:]
        end
    end
end

# innovation_λ(m::Model, x, r, obs_mean, obs_variance_over_K, finite) = all(finite) ? m.d_observation(x,r) .* (m.observation(x,r) .- obs_mean)./obs_variance_over_K : zeros(x)
#
# @views function innovation_dλ!(innovation, m::Model{N,L,R,U}, t, x, p, dx, dp, Nobs, K_over_obs_variance, x_minus_mean_obs, δobs, x_minus_mean_obs_times_δobs, finite) where {N,L,R,U}
#     fill!(innovation, 0.)
#     m.observation!(m, t, x, p)
#     m.observation_jacobianx!(m, t, x, p)
#     m.observation_hessianxx!(m, t, x, p)
#     m.observation_hessianxr!(m, t, x, p)
#     for _j in 1:U
#         if finite[_j]
#             innovation .+= K_over_obs_variance[_j] .* ( δobs[_j] .* m.observation_jacobianx[_j,:] .+ x_minus_mean_obs[_j] .* (m.observation_hessianxx[_j,:,:] * dx .+ m.observation_hessianxr[_j,:,:] * dp[L-N+1:L-N+R]) .- K_over_obs_variance[_j] * 2./Nobs[_j] * δobs[_j] .* m.observation_jacobianx[_j,:] .* x_minus_mean_obs_times_δobs[_j])
#         end
#     end
# end
#
# @views function innovation_dν!(innovation, m::Model{N,L,R,U}, t, x, p, dx, dp, Nobs, K_over_obs_variance, x_minus_mean_obs, δobs, x_minus_mean_obs_times_δobs, finite) where {N,L,R,U}
#     fill!(innovation, 0.)
#     m.observation!(m, t, x, p)
#     m.observation_jacobianp!(m, t, x, p)
#     m.observation_hessianpp!(m, t, x, p)
#     for _j in 1:U
#         if finite[_j]
#             innovation .+= K_over_obs_variance[_j] .* ( δobs[_j] .* m.observation_jacobianp[_j,:] .+ x_minus_mean_obs[_j] .* (m.observation_hessianpp[_j,:,:] * dp[1:L-N]) .- K_over_obs_variance[_j] * 2./Nobs[_j] * δobs[_j] .* m.observation_jacobianp[_j,:] .* x_minus_mean_obs_times_δobs[_j])
#         end
#     end
# end
#
# @views function innovation_dμ!(innovation, m::Model{N,L,R,U}, t, x, p, dx, dp, Nobs, K_over_obs_variance, x_minus_mean_obs, δobs, x_minus_mean_obs_times_δobs, finite) where {N,L,R,U}
#     fill!(innovation, 0.)
#     m.observation!(m, t, x, p)
#     m.observation_jacobianr!(m, t, x, p)
#     m.observation_hessianxx!(m, t, x, p)
#     m.observation_hessianxr!(m, t, x, p)
#     for _j in 1:U
#         if finite[_j]
#             innovation .+= K_over_obs_variance[_j] .* ( δobs[_j] .* m.observation_jacobianr[_j,:] .+ x_minus_mean_obs[_j] .* (m.observation_hessianrr[_j,:,:] * dp[L-N+1:L-N+R] .+ m.observation_hessianxr[_j,:,:]' * dx) .- K_over_obs_variance[_j] * 2./Nobs[_j] * δobs[_j] .* m.observation_jacobianr[_j,:] .* x_minus_mean_obs_times_δobs[_j])
#         end
#     end
# end

# @views innovation_dλ(m::Model, x_i, dx_i, Nobs, K_over_obs_variance, x_minus_mean_obs_i, x_minus_mean_obs_times_dx, finite_i) = finite_i ? K_over_obs_variance * ( ( m.dd_observation(x_i) * x_minus_mean_obs_i + m.d_observation(x_i)^2 ) * dx_i - K_over_obs_variance * oftype(K_over_obs_variance, 2.) / Nobs * m.d_observation(x_i) * x_minus_mean_obs_i * x_minus_mean_obs_times_dx) : zero(K_over_obs_variance)

@views function next_x_first!(a::Adjoint{N,L,R,U,K,T},       m::Model,       t, x_prev, x) where {N,L,R,U,K,T}
    m.dxdt!(m, t, x_prev, a.p)
    x .= x_prev .+ m.dxdt .* a.dt .+ dot(sqrt.(inv.(diag(a.sys_precision))), randn(N)) .* a.dt
    nothing
end

@views function orbit_first!(a::Adjoint{N,L,R,U,K,T}, m) where {N,L,R,U,K,T<:AbstractFloat}
    for _i in 1:a.steps
        next_x_first!(a, m, a.dt*(_i-1), a.x[:,_i], a.x[:,_i+1]) # Use next step's t
    end
    nothing
end

@views function orbit!(a::Adjoint{N,L,R,U,K,T}, m, obs_mean) where {N,L,R,U,K,T<:AbstractFloat}
    noise = zeros(T, N)
    for _i in 1:a.steps
        next_x!(a, m, a.dt*(_i-1), a.x[:,_i], a.x[:,_i+1], noise, obs_mean[:,_i]) # Use next step's t
    end
    nothing
end

#
# @views function neighboring!(a, m)
#     for _i in 1:a.steps
#         # next_dx!(a, m, a.dt*(_i-1), a.x[:,_i],            a.dx[:,_i], a.dx[:,_i+1]) # bug!
#         next_dx!(a, m, a.dt*_i, a.x[:,_i+1],            a.dx[:,_i], a.dx[:,_i+1]) # bug fixed
#     end
#     nothing
# end
#
# @views function obs_variance!(a::Adjoint{N,L,R,U,K}, m::Model{N,L,R,U}) where {N,L,R,U,K}
#     fill!(a.obs_variance, 0.)
#
#     for _t in m.time_point
#         _j = Int(_t/a.dt) + 1
#         m.observation!(m, _t, a.x[:,_j], a.p)
#         a.obs_variance .+= abs2.(m.observation .- a.obs_mean[:,_j])
#     end
#
#     a.obs_variance .= (a.pseudo_obs_TSS .+ K .* (a.obs_variance .+ a.obs_filterd_var) ) ./ a.Nobs
#
#     # a.obs_variance[1] = (a.pseudo_obs_TSS[1] + K * (mapreduce(abs2, +, ([m.observation(a.x[:,_j], a.r)[1] for _j in 1:a.steps+1] .- a.obs_mean[1,:])[a.finite[1,:]]) + a.obs_filterd_var[1])) / a.Nobs[1]
#     # a.obs_variance[N+1] = (a.pseudo_obs_TSS[N+1] + K * (mapreduce(abs2, +, exp.(a.p[2:end]) .- a.obs_mean[N+1,:][a.finite[N+1,:]]) + a.obs_filterd_var[N+1])) / a.Nobs[N+1]
#     nothing
# end

@views function gradient!(a::Adjoint{N,L,R,U,K}, m::Model{N,L,R,U}, obs_mean, gr) where {N,L,R,U,K} # assuming x[:,1] .= x0; p .= p; orbit!(dxdt, t, x, p, dt); is already run. λ[:,1] is the gradient.
    m.observation!(m, a.t[end], a.x[:,end], a.r)
    m.observation_jacobianx!(m, a.t[end], a.x[:,end], a.r)
    m.jacobianx!(m, a.t[a.steps], a.x[:,a.steps], a.p)
    a.λ[:,end] .= a.sys_precision * m.jacobianx \ (a.sys_precision * (a.x[:,end] .- a.x[:,a.steps] .- a.dxdt .* a.dt) .+ m.observation_jacobianx' * a.obs_precision * (m.observation .- obs_mean[end]) .* a.dt^2) #<- implemented until here

    m.calc_eqdot!(m, a.p[2:4])
    obs_variance_over_K = a.obs_variance ./ K
    prev_λ!(a, m, a.t[end], a.x[:,end], zeros(L+R), a.λ[:,end], a.obs_mean[:,end], obs_variance_over_K, a.finite[:,end])
    for _i in a.steps:-1:2
        prev_λ!(a, m, a.t[_i], a.x[:,_i], a.λ[:,_i+1], a.λ[:,_i], a.obs_mean[:,_i], obs_variance_over_K, a.finite[:,_i])
    end

    innovation_λ!(gr[1:N], m, a.t[1], a.x[:,1], a.p, a.obs_mean[:,1], obs_variance_over_K, a.finite[:,1])
    gr[1:N]     .+= a.λ[1:N,2]
    # gr[N+1]      .= a.λ[N+1,2]   .+ a.regularization .* a.p[1]
    # gr[N+2:L]    .= a.λ[N+2:L,2] .+ (exp.(a.p[2:4]) .- a.obs_mean[N+1,:][a.finite[N+1,:]]) ./ obs_variance_over_K[N+1] .* exp.(a.p[2:4])

    innovation_ν!(a.λ[N+1:L,1], m, a.t[1], a.x[:,1], a.p, a.obs_mean[:,1], obs_variance_over_K, a.finite[:,1])
    a.λ[N+1:L,1] .+= a.λ[N+1:L,2]
    a.λ[N+1,1] += a.regularization * a.p[1]
    gr[N+1:L]  .= a.λ[N+1:L,1]

    innovation_μ!(a.λ[L+1:L+R,1], m, a.t[1], a.x[:,1], a.p, a.obs_mean[:,1], obs_variance_over_K, a.finite[:,1])
    a.λ[L+1:L+R,1] .+= a.λ[L+1:L+R,2]
    gr[L+1:L+R]     .= a.λ[L+1:L+R,1]
    #
    # fill!(gr[L+1:L+R], 0.)
    # # for _j in 1:a.steps
    # #     if a.finite[1,_j]
    # #         m.dr_observation!(m, a.x[:,_j], a.r)
    # #         gr[L+1:L+R] .+= (observation(a.x[:,_j], a.r)[1] - a.obs_mean[1,_j]) / obs_variance_over_K[1] .* m.dr_observation[1,:]
    # #     end
    # # end
    #
    # for _j in 1:(a.steps+1)
    #     if a.finite[1,_j]
    #         gr[L+1] += (a.r[1] + a.r[2]*exp(a.x[1,_j]) - a.obs_mean[1,_j]) / obs_variance_over_K[1]
    #         gr[L+2] += (a.r[1] + a.r[2]*exp(a.x[1,_j]) - a.obs_mean[1,_j]) * exp(a.x[1,_j])/ obs_variance_over_K[1]
    #     end
    # end

    nothing
end

@views function hessian_vector_product!(a::Adjoint{N,L,R,U,K}, m::Model{N,L,R,U}, x_minus_mean_obs, δobs, x_minus_mean_obs_times_δobs, hv) where {N,L,R,U,K} # assuming dx[:,1] .= dx0; dp .= dp; neighboring!(jacobian, dt, t, x, p, dx, dp); is already run. dλ[:,1] is the hessian_vector_product.
    K_over_obs_variance = K ./ a.obs_variance
    prev_dλ!(a, m, a.t[end], a.x[:,end], a.dx[:,end], a.dp, a.λ[:,end], zeros(L+R), a.dλ[:,end], a.Nobs, K_over_obs_variance, x_minus_mean_obs[:,end], δobs[:,end], x_minus_mean_obs_times_δobs, a.finite[:,end])
    for _i in a.steps:-1:2
        prev_dλ!(a, m, a.t[_i], a.x[:,_i], a.dx[:,_i], a.dp, a.λ[:,_i], a.dλ[:,_i+1], a.dλ[:,_i], a.Nobs, K_over_obs_variance, x_minus_mean_obs[:,_i], δobs[:,_i], x_minus_mean_obs_times_δobs, a.finite[:,_i])
    end

    innovation_dλ!(hv[1:N], m, a.t[1], a.x[:,1], a.p, a.dx[:,1], a.dp, a.Nobs, K_over_obs_variance, x_minus_mean_obs[:,1], δobs[:,1], x_minus_mean_obs_times_δobs, a.finite[:,1])
    hv[1:N]     .+= a.dλ[1:N,2]

    innovation_dν!(a.dλ[N+1:L,1], m, a.t[1], a.x[:,1], a.p, a.dx[:,1], a.dp, a.Nobs, K_over_obs_variance, x_minus_mean_obs[:,1], δobs[:,1], x_minus_mean_obs_times_δobs, a.finite[:,1])
    a.dλ[N+1:L,1] .+= a.dλ[N+1:L,2]
    a.dλ[N+1,1] += a.regularization * a.dp[1]
    hv[N+1:L] .= a.dλ[N+1:L,1]
    # hv[N+2:L]   .= a.dλ[N+2:L,2] .+ K_over_obs_variance[N+1] * ( (2. .* exp.(a.p[2:end]) .- a.obs_mean[N+1,:][a.finite[N+1,:]]) .* exp.(a.p[2:end]) .* a.dp[2:end] .- K_over_obs_variance[N+1] * oftype(K_over_obs_variance[N+1], 2.) / a.Nobs[N+1] * u_minus_mean_obs * u_minus_mean_obs_times_du)

    innovation_dμ!(a.dλ[L+1:L+R,1], m, a.t[1], a.x[:,1], a.p, a.dx[:,1], a.dp, a.Nobs, K_over_obs_variance, x_minus_mean_obs[:,1], δobs[:,1], x_minus_mean_obs_times_δobs, a.finite[:,1])
    a.dλ[L+1:L+R,1] .+= a.dλ[L+1:L+R,2]
    hv[L+1:L+R] .= a.dλ[L+1:L+R,1]
    nothing
end

@views function cost(a::Adjoint{N,L,R,U,K}) where {N,L,R,U,K} # assuming x[:,1] .= x0; orbit!(dxdt, t, x, p, dt); is already run
    c = a.regularization * a.p[1] * a.p[1]
    for _i in 1:U
        if a.Nobs[_i] > 0
            c += a.Nobs[_i] * log(a.obs_variance[_i])
        end
    end
    c / oftype(a.dt, 2.)
end

@views function initialize!(a::Adjoint{N,M,R}, θ) where {N,M,R}
    copy!(a.x[:,1], θ[1:N])
    copy!(a.p, θ[N+1:N+M])
    copy!(a.r, θ[N+M+1:end])
    # copy!(a.r, θ[L+1:end])
    nothing
end

# @views function initialize_p!(a::Adjoint{N}, p) where {N}
#     copy!(a.p, p)
#     nothing
# end

@views function inv_j!(a::Adjoint{N}, m, t, x) where {N}
    m.jacobianx!(m, t, x, a.p)
    a.jacobian_inv .= eye(N) .- m.jacobianx .* a.dt
    nothing
end

@views function residual!(a::Adjoint, m::Model, t, x_prev, x)
    m.dxdt!(m, t, x, a.p)
    a.res .= x .- x_prev .- m.dxdt .* a.dt
    nothing
end

@views function next_x!(a::Adjoint{N,L,R,U,K,T},       m::Model,       t, x_prev, x, noise, obs_mean_prev) where {N,L,R,U,K,T}
    m.dxdt!(m, t, x_prev, a.p) # forward Euler step as first guess
    m.jacobianx!(m, t, x_prev, a.p)
    m.observation!(m, t, x_prev, a.r)
    m.observation_jacobianx!(m, t, x_prev, a.r)
    # noise .= (I + m.jacobianx .* a.dt)' * a.sys_precision \ ( a.sys_precision * noise + m.observation_jacobianx' * a.obs_precision * (m.observation .- obs_mean_prev) .* a.dt^2 )
    # # println("noise: $noise")
    # x .= x_prev .+ m.dxdt .* a.dt .+ noise  #<- editing this line

    noise[1] = (noise[1] + inv(a.sys_precision[1,1]) * a.obs_precision[1,1] * a.dt^2 * (a.r[1] + x_prev[1] - obs_mean_prev[1]))/(1. - a.p[1]*a.dt)
    x[1] = (1. - a.p[1]*a.dt) * x_prev[1] + noise[1]
end

@views function next_dx!(a::Adjoint{N,L,R,U,K,T},       m::Model,       t, x,        dx, dx_nxt) where {N,L,R,U,K,T}
    inv_j!(a, m, t, x)
    m.jacobianp!(m, t, x, a.p)
    dx_nxt .= a.jacobian_inv \ ( dx .+  m.jacobianp * a.dp[1:L-N] .* a.dt )
    nothing
end

@views function prev_λ!( a::Adjoint{N,L,R,U},    m::Model{N,L,R,U},    t, x,                   λ, λ_prev, obs_mean, obs_variance_over_K, finite) where {N,L,R,U}
    inv_j!(a, m, t, x)
    m.jacobianp!(m, t, x, a.p)

    innovation_λ!(λ_prev[1:N], m, t, x, a.p, obs_mean, obs_variance_over_K, finite)
    λ_prev[1:N] .= a.jacobian_inv' \ (λ[1:N] .+ λ_prev[1:N])

    innovation_ν!(λ_prev[N+1:L], m, t, x, a.p, obs_mean, obs_variance_over_K, finite)
    λ_prev[N+1:L] .+= λ[N+1:L] + m.jacobianp' * λ_prev[1:N] .* a.dt

    innovation_μ!(λ_prev[L+1:L+R], m, t, x, a.p, obs_mean, obs_variance_over_K, finite)
    λ_prev[L+1:L+R] .+= λ[L+1:L+R]

    nothing
end

@views function prev_dλ!(a::Adjoint{N, L, R}, m::Model{N, L, R}, t, x,        dx,   dp,     λ,       dλ, dλ_prev, Nobs, K_over_obs_variance, x_minus_mean_obs, δobs, x_minus_mean_obs_times_δobs, finite) where {N, L, R}
    M = L-N
    inv_j!(a, m, t, x)
    m.jacobianp!(m, t, x, a.p)
    m.hessianxx!(m, t, x, a.p)
    m.hessianxp!(m, t, x, a.p)
    m.hessianpp!(m, t, x, a.p)

    innovation_dλ!(dλ_prev[1:N], m, t, x, a.p, dx, dp, Nobs, K_over_obs_variance, x_minus_mean_obs, δobs, x_minus_mean_obs_times_δobs, finite)
    dλ_prev[1:N] .= a.jacobian_inv' \ ( dλ[1:N] .+ ( reshape( reshape(m.hessianxx, N*N, N) * dx, N, N ) .+ reshape( reshape(m.hessianxp, N*N, M) * a.dp[1:M], N, N ) )' * λ[1:N] .* a.dt .+ dλ_prev[1:N] )

    innovation_dν!(dλ_prev[N+1:L], m, t, x, a.p, dx, dp, Nobs, K_over_obs_variance, x_minus_mean_obs, δobs, x_minus_mean_obs_times_δobs, finite)
    # dλ_prev[N+1:end] .= dλ[N+1:end] .+ ( ( reshape( reshape(permutedims(m.hessianxp, [1,3,2]), N*M, N) * dx, N, M ) .+ reshape( reshape(m.hessianpp, N*M, M) * a.dp, N, M ) )' * λ[1:N] .+ m.jacobianp' * dλ[1:N] ) .* a.dt # bug!
    dλ_prev[N+1:L] .+= dλ[N+1:L] .+ ( ( reshape( reshape(permutedims(m.hessianxp, [1,3,2]), N*M, N) * dx, N, M ) .+ reshape( reshape(m.hessianpp, N*M, M) * a.dp[1:M], N, M ) )' * λ[1:N] .+ m.jacobianp' * dλ_prev[1:N] ) .* a.dt # bug fixed

    innovation_dμ!(dλ_prev[L+1:L+R], m, t, x, a.p, dx, dp, Nobs, K_over_obs_variance, x_minus_mean_obs, δobs, x_minus_mean_obs_times_δobs, finite)
    dλ_prev[L+1:L+R] .+= dλ[L+1:L+R]
    nothing
end

@views function fg!(F, ∇θ, θ, a::Adjoint{N,L,R}, m::Model{N,L,R}) where {N,L,R}
    initialize!(a, θ)
    orbit!(a, m)
    if !(∇θ == nothing)
        gradient!(a, m, ∇θ)
    end
    if !(F == nothing)
        F = cost(a)
    end
end

# function minimize!(initial_θ, a, m)
#     df = NLSolversBase.OnceDifferentiable(NLSolversBase.only_fg!((F, ∇θ, θ) -> fg!(F, ∇θ, θ, a, m)), initial_θ)
#     Optim.optimize(df, initial_θ, LBFGS())
# end

@views function minimize!(initial_θ, a::Adjoint{N,L,R,U,K,T}, m::Model{N,L,R,U,T}) where {N,L,R,U,K,T<:AbstractFloat} # Fixed. views is definitely needed for copy!
    initialize!(a, initial_θ)
    orbit!(a, m)
    F = cost(a)

    ∇θ = Vector{T}(L+R)
    gradient!(a, m, ∇θ)

    df = OnceDifferentiable(θ -> fg!(F, nothing, θ, a, m), (∇θ, θ) -> fg!(nothing, ∇θ, θ, a, m), (∇θ, θ) -> fg!(F, ∇θ, θ, a, m), initial_θ, F, ∇θ, inplace=true)
    # options = Optim.Options(;x_tol=1e-32, f_tol=1e-32, g_tol=1e-8, iterations=1_000, store_trace=true, show_trace=false, show_every=1)
    options = Optim.Options(;x_tol=1e-32, f_tol=1e-32, g_tol=1e-8, iterations=1_000, store_trace=true, show_trace=false, show_every=1)
    lbfgs_ls_scaled_hz = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true), linesearch = LineSearches.HagerZhang())
    # lbfgs_ls_alpha_hz = LBFGS(;alphaguess = LineSearches.InitialStatic(;alpha=0.001), linesearch = LineSearches.HagerZhang())
    # optimize(df, initial_p, lbfgs_ls_alpha_hz, options)
    optimize(df, initial_θ, lbfgs_ls_scaled_hz, options)
end

nanzero(x) = isnan(x) ? zero(x) : x

@views function _covariance!(a::Adjoint{N,L,R,U,K,T}, m::Model{N,L,R,U,T}, x_minus_mean_obs, δobs, x_minus_mean_obs_times_δobs, hv) where {N,L,R,U,K,T<:AbstractFloat}
    neighboring!(a, m)
    for _t in m.time_point
        _j = Int(_t/a.dt) + 1
        m.observation!(m, _t, a.x[:,_j], a.p)
        m.observation_jacobianx!(m, _t, a.x[:,_j], a.p)
        m.observation_jacobianp!(m, _t, a.x[:,_j], a.p)
        m.observation_jacobianr!(m, _t, a.x[:,_j], a.p)
        x_minus_mean_obs[:,_j] = m.observation .- a.obs_mean[:,_j]
        δobs[:,_j] = m.observation_jacobianx * a.dx[:,_j] .+ m.observation_jacobianp * a.dp[1:L-N] .+ m.observation_jacobianr * a.dp[L-N+1:L-N+R]
    end
    for _j in 1:U
        x_minus_mean_obs_times_δobs[_j] .= mapreduce(nanzero, +, x_minus_mean_obs[_j,:] .* δobs[_j,:])
    end
    hessian_vector_product!(a, m, x_minus_mean_obs, δobs, x_minus_mean_obs_times_δobs, hv)
end

@views function covariance!(a::Adjoint{N,L,R,U,K,T}, m::Model{N,L,R,U,T}) where {N,L,R,U,K,T<:AbstractFloat}
    fill!(a.dx[:,1], 0.)
    fill!(a.dp, 0.)
    hessian = Matrix{T}(L+R,L+R)

    x_minus_mean_obs = Matrix{T}(U,a.steps+1)
    δobs = Matrix{T}(U,a.steps+1)
    x_minus_mean_obs_times_δobs = Vector{T}(U)
    fill!(x_minus_mean_obs, NaN)
    fill!(δobs, NaN)

    # u_minus_mean_obs = (exp.(a.p[2:end]) .- a.obs_mean[N+1,:][a.finite[N+1,:]]) .* exp.(a.p[2:end])

    # h11 = K/a.obs_variance[1] * (1. + 1./(1. + a.dt * a.p[1])^2. - ( a.x[1,1] - a.obs_mean[1,1] + (a.x[1,2] - a.obs_mean[1,2]) / (1. + a.dt * a.p[1]) )^2. / a.obs_variance[1] )
    # h22 = K/a.obs_variance[1] * (a.dt / (1. + a.dt * a.p[1]))^2. * a.x[1,2] * (3a.x[1,2] - 2a.obs_mean[1,2] - a.x[1,2] * x_minus_mean_obs[1,2]^2. / a.obs_variance[1]) + 1.
    # h33 = K/a.obs_variance[2] * exp(a.p[2]) * (2exp(a.p[2]) - a.obs_mean[2,1] - exp(a.p[2]) * (exp(a.p[2]) - a.obs_mean[2,1])^2. / a.obs_variance[2])
    # h44 = K/a.obs_variance[2] * exp(a.p[2]) * (2exp(a.p[2]) - a.obs_mean[2,1] - exp(a.p[2]) * (exp(a.p[2]) - a.obs_mean[2,1])^2. / a.obs_variance[2]) + K/a.obs_variance[1] * a.dt / (1. + a.dt*a.p[1]) * exp(a.p[3]) * ( a.x[1,2] - a.obs_mean[1,2] + a.dt / (1. + a.dt*a.p[1]) * exp(a.p[3]) * (1. - (a.x[1,2] - a.obs_mean[1,2])^2. / a.obs_variance[1]) )

    # h12 = -a.dt/(1.+a.dt*a.p[1]) * ( K/a.obs_variance[1] * x_minus_mean_obs[1,2] + a.x[2] * ( 1./(1.+a.dt*a.p[1]) * K/a.obs_variance[1] * (1. - x_minus_mean_obs[2]^2/a.obs_variance[1]) - K/a.obs_variance[1]^2 * x_minus_mean_obs[1] * x_minus_mean_obs[2] ) )
    # h14 = (1./(1.+a.dt*a.p[1]) * K/a.obs_variance[1] * (1. - x_minus_mean_obs[1,2]^2/a.obs_variance[1]) - K/a.obs_variance[1]^2 * x_minus_mean_obs[1,1] * x_minus_mean_obs[1,2]) * a.dt / (1. + a.dt * a.p[1]) * exp(a.p[3])

    for i in 1:N
        a.dx[i,1] = 1.
        _covariance!(a, m, x_minus_mean_obs, δobs, x_minus_mean_obs_times_δobs, hessian[:,i])
        a.dx[i,1] = 0.
    end
    for i in N+1:L+R
        a.dp[i-N] = 1.
        _covariance!(a, m, x_minus_mean_obs, δobs, x_minus_mean_obs_times_δobs, hessian[:,i])
        a.dp[i-N] = 0.
    end
    θ = vcat(a.x[:,1], a.p)

    # println("hessian12_alg: $(hessian[1,2])\t$(hessian[2,1])")
    # println("hessian12_dir: $(h12)")
    # println("hessian14_alg: $(hessian[1,4])\t$(hessian[4,1])")
    # println("hessian14_dir: $(h14)")

    covariance = nothing
    try
        covariance = inv(hessian)
    catch message
        println(STDERR, "CI calculation failed.\nReason: Hessian inversion failed due to $message")
        return AssimilationResults(θ, a.obs_variance, hessian)
    end
    stddev = nothing
    try
        stddev = sqrt.(diag(covariance))
    catch message
        println("CI calculation failed.")
        # println("hessian")
        # println(hessian)
        # println("covariance:")
        # println(covariance)
        if (minimum(diag(covariance)) < 0)
            println(STDERR, "Reason: Negative variance!")
        else
            println(STDERR, "Reason: Taking sqrt of variance failed due to $message")
        end
        return AssimilationResults(θ, a.obs_variance, hessian, covariance)
    end
    return AssimilationResults(θ, a.obs_variance, stddev, hessian, covariance)
end
