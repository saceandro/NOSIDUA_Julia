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

@views function innovation_μ!(innovation, m::Model{N,L,R,U}, t, x, p, obs_mean, obs_variance_over_K, finite) where {N,L,R,U}
    fill!(innovation, 0.)
    m.observation!(m, t, x, p)
    m.observation_jacobianr!(m, t, x, p)
    for _j in 1:U
        if finite[_j]
            innovation .+= (m.observation[_j] - obs_mean[_j]) / obs_variance_over_K[_j] .* m.observation_jacobianr[_j,:]
        end
    end
end

# innovation_λ(m::Model, x, r, obs_mean, obs_variance_over_K, finite) = all(finite) ? m.d_observation(x,r) .* (m.observation(x,r) .- obs_mean)./obs_variance_over_K : zeros(x)

@views innovation_dλ(m::Model, x_i, dx_i, Nobs, K_over_obs_variance, x_minus_mean_obs_i, x_minus_mean_obs_times_dx, finite_i) = finite_i ? K_over_obs_variance * ( ( m.dd_observation(x_i) * x_minus_mean_obs_i + m.d_observation(x_i)^2 ) * dx_i - K_over_obs_variance * oftype(K_over_obs_variance, 2.) / Nobs * m.d_observation(x_i) * x_minus_mean_obs_i * x_minus_mean_obs_times_dx) : zero(K_over_obs_variance)

@views function orbit_first!(a::Adjoint{N,L,R,U,K,T}, m) where {N,L,R,U,K,T<:AbstractFloat}
    # res = zeros(N)
    # x_min = zeros(N)
    # res_min = zeros(N)
    m.calc_eqdot!(m, a.p[2:4])
    for _i in 1:a.steps
        next_x!(a, m, a.dt*_i, a.x[:,_i], a.x[:,_i+1]) # Use next step's t
    end
    nothing
end

@views function orbit!(a::Adjoint{N,L,R,U,K,T}, m) where {N,L,R,U,K,T<:AbstractFloat}
    # res = zeros(N)
    # x_min = zeros(N)
    # res_min = zeros(N)
    m.calc_eqdot!(m, a.p[2:4])
    for _i in 1:a.steps
        next_x!(a, m, a.dt*_i, a.x[:,_i], a.x[:,_i+1]) # Use next step's t
    end
    obs_variance!(a, m)
    nothing
end

@views function neighboring!(a, m)
    for _i in 1:a.steps
        # next_dx!(a, m, a.dt*(_i-1), a.x[:,_i],            a.dx[:,_i], a.dx[:,_i+1]) # bug!
        next_dx!(a, m, a.dt*_i, a.x[:,_i+1],            a.dx[:,_i], a.dx[:,_i+1]) # bug fixed
    end
    nothing
end

@views function obs_variance!(a::Adjoint{N,L,R,U,K}, m::Model{N,L,R,U}) where {N,L,R,U,K}
    fill!(a.obs_variance, 0.)

    for _t in m.time_point
        _j = Int(_t/a.dt) + 1
        m.observation!(m, _t, a.x[:,_j], a.p)
        a.obs_variance .+= abs2.(m.observation .- a.obs_mean[:,_j])
    end

    a.obs_variance .= (a.pseudo_obs_TSS .+ K .* (a.obs_variance .+ a.obs_filterd_var) ) ./ a.Nobs

    # a.obs_variance[1] = (a.pseudo_obs_TSS[1] + K * (mapreduce(abs2, +, ([m.observation(a.x[:,_j], a.r)[1] for _j in 1:a.steps+1] .- a.obs_mean[1,:])[a.finite[1,:]]) + a.obs_filterd_var[1])) / a.Nobs[1]
    # a.obs_variance[N+1] = (a.pseudo_obs_TSS[N+1] + K * (mapreduce(abs2, +, exp.(a.p[2:end]) .- a.obs_mean[N+1,:][a.finite[N+1,:]]) + a.obs_filterd_var[N+1])) / a.Nobs[N+1]
    nothing
end

@views function gradient!(a::Adjoint{N,L,R,U,K}, m::Model{N,L,R,U}, gr) where {N,L,R,U,K} # assuming x[:,1] .= x0; p .= p; orbit!(dxdt, t, x, p, dt); is already run. λ[:,1] is the gradient.
    m.calc_eqdot!(m, a.p[2:4])
    obs_variance_over_K = a.obs_variance ./ K
    prev_λ!(a, m, a.t[end], a.x[:,end], zeros(L+R), a.λ[:,end], a.obs_mean[:,end], obs_variance_over_K, a.finite[:,end])
    for _i in a.steps:-1:2
        prev_λ!(a, m, a.t[_i], a.x[:,_i], a.λ[:,_i+1], a.λ[:,_i], a.obs_mean[:,_i], obs_variance_over_K, a.finite[:,_i])
    end

    innovation_λ!(gr[1:N], m, a.t[1], a.x[:,1], a.p, a.obs_mean[:,1], obs_variance_over_K, a.finite[:,1])
    gr[1:N]     .+= a.λ[1:N,2]
    gr[N+1]      .= a.λ[N+1,2]   .+ a.regularization .* a.p[1]
    gr[N+2:L]    .= a.λ[N+2:L,2] .+ (exp.(a.p[2:4]) .- a.obs_mean[N+1,:][a.finite[N+1,:]]) ./ obs_variance_over_K[N+1] .* exp.(a.p[2:4])

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

@views function hessian_vector_product!(a::Adjoint{N,L,R,U,K}, m::Model{N,L,R,U}, x_minus_mean_obs, u_minus_mean_obs, x_minus_mean_obs_times_dx, u_minus_mean_obs_times_du, hv) where {N,L,R,U,K} # assuming dx[:,1] .= dx0; dp .= dp; neighboring!(jacobian, dt, t, x, p, dx, dp); is already run. dλ[:,1] is the hessian_vector_product.
    K_over_obs_variance = K ./ a.obs_variance
    prev_dλ!(a, m, a.t[end], a.x[:,end], a.dx[:,end], a.λ[:,end], zeros(L), a.dλ[:,end], a.Nobs[1:N], K_over_obs_variance[1:N], x_minus_mean_obs[:,end], x_minus_mean_obs_times_dx, a.finite[1:N,end])
    for _i in a.steps:-1:2
        prev_dλ!(a, m, a.t[_i], a.x[:,_i], a.dx[:,_i], a.λ[:,_i], a.dλ[:,_i+1], a.dλ[:,_i], a.Nobs[1:N], K_over_obs_variance[1:N], x_minus_mean_obs[:,_i], x_minus_mean_obs_times_dx, a.finite[1:N,_i])
    end
    hv[1:N]     .= a.dλ[1:N,2] .+ innovation_dλ.(m, a.x[:,1], a.dx[:,1], a.Nobs[1:N], K_over_obs_variance[1:N], x_minus_mean_obs[:,1], x_minus_mean_obs_times_dx, a.finite[1:N,1])
    # println("hv1_prev: $(a.dλ[1:N,2] .+ innovation_dλ.(m, a.x[:,1], a.dx[:,1], a.Nobs[1:N], K_over_obs_variance[1:N], x_minus_mean_obs[:,1], x_minus_mean_obs_times_dx, a.finite[1:N,1]))")
    # hv[1]        = a.dλ[1,2] .+ K_over_obs_variance[1] * (a.dx[1,1] - K_over_obs_variance[1] * 2./a.Nobs[1] * x_minus_mean_obs[1,1] * x_minus_mean_obs_times_dx[1])
    # println("hv1_curr: $(hv[1])")
    hv[N+1]     .= a.dλ[N+1,2] .+ a.regularization .* a.dp[1]
    hv[N+2:L]   .= a.dλ[N+2:L,2] .+ K_over_obs_variance[N+1] * ( (2. .* exp.(a.p[2:end]) .- a.obs_mean[N+1,:][a.finite[N+1,:]]) .* exp.(a.p[2:end]) .* a.dp[2:end] .- K_over_obs_variance[N+1] * oftype(K_over_obs_variance[N+1], 2.) / a.Nobs[N+1] * u_minus_mean_obs * u_minus_mean_obs_times_du)
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

@views function initialize!(a::Adjoint{N,L,R}, θ) where {N,L,R}
    copy!(a.x[:,1], θ[1:N])
    copy!(a.p, θ[N+1:L+R])
    # copy!(a.r, θ[L+1:end])
    nothing
end

# @views function initialize_p!(a::Adjoint{N}, p) where {N}
#     copy!(a.p, p)
#     nothing
# end

@views function inv_j!(a::Adjoint{N}, m, t, x) where {N}
    m.jacobianx!(m, t, x, a.p)
    a.jacobian_inv .= inv( eye(N) .- m.jacobianx .* a.dt )
    nothing
end

@views function residual!(a::Adjoint, m::Model, t, x_prev, x)
    m.dxdt!(m, t, x, a.p)
    a.res .= x .- x_prev .- m.dxdt .* a.dt
    nothing
end

@views function next_x!(a::Adjoint{N,L,R,U,K,T},       m::Model,       t, x_prev, x) where {N,L,R,U,K,T}
    m.dxdt!(m, t, x_prev, a.p) # forward Euler step as first guess
    x .= x_prev .+ m.dxdt .* a.dt
    residual!(a, m, t, x_prev, x)

    # norm_res_min = norm(res, 2)
    # res_min .= res
    # x_min .= x

    # if norm(res, 2) < tol
    if norm(a.res, 2) < a.newton_tol
        return nothing
    end

    for i in 1:a.trylimit
        m.jacobianx!(m, t, x, a.p)
        x .-= (I - m.jacobianx .* a.dt) \ a.res
        residual!(a, m, t, x_prev, x)
        # norm_res = norm(res, 2)
        # if norm_res < norm_res_min
        #     norm_res_min = norm_res
        #     res_min .= res
        #     x_min .= x
        # end
        # if norm(res, 2)<tol
        if norm(a.res, 2) < a.newton_tol
            # println("newton x: $(x[1])")
            # println("true   x: $( (x_prev[1] + a.dt * exp(a.p[3]))/(1.+a.dt*a.p[1]) )")
            return nothing
        end
    end

    error("Newton's method did not converged!")
end

@views function next_dx!(a::Adjoint,       m::Model,       t, x,        dx, dx_nxt)
    inv_j!(a, m, t, x)
    m.jacobianp!(m, t, x, a.p)
    dx_nxt .= a.jacobian_inv * ( dx .+  m.jacobianp * a.dp .* a.dt )
    nothing
end

@views function prev_λ!( a::Adjoint{N,L,R,U},    m::Model{N,L,R,U},    t, x,                   λ, λ_prev, obs_mean, obs_variance_over_K, finite) where {N,L,R,U}
    inv_j!(a, m, t, x)
    m.jacobianp!(m, t, x, a.p)

    innovation_λ!(λ_prev[1:N], m, t, x, a.p, obs_mean, obs_variance_over_K, finite)

    λ_prev[1:N] .= a.jacobian_inv' * (λ[1:N] .+ λ_prev[1:N])
    λ_prev[N+1:L] .= λ[N+1:L] + m.jacobianp' * λ_prev[1:N] .* a.dt

    innovation_μ!(λ_prev[L+1:L+R], m, t, x, a.p, obs_mean, obs_variance_over_K, finite)
    λ_prev[L+1:L+R] .+= λ[L+1:L+R]

    nothing
end

@views function prev_dλ!(a::Adjoint{N, L}, m::Model{N, L}, t, x,        dx,        λ,       dλ, dλ_prev, Nobs, K_over_obs_variance, x_minus_mean_obs, x_minus_mean_obs_times_dx, finite) where {N, L}
    M = L-N
    inv_j!(a, m, t, x)
    m.jacobianp!(m, t, x, a.p)
    m.hessianxx!(m, t, x, a.p)
    m.hessianxp!(m, t, x, a.p)
    m.hessianpp!(m, t, x, a.p)
    dλ_prev[1:N] .= a.jacobian_inv' * ( dλ[1:N] .+ ( reshape( reshape(m.hessianxx, N*N, N) * dx, N, N ) .+ reshape( reshape(m.hessianxp, N*N, M) * a.dp, N, N ) )' * λ[1:N] .* a.dt .+ innovation_dλ.(m, x, dx, Nobs, K_over_obs_variance, x_minus_mean_obs, x_minus_mean_obs_times_dx, finite) )
    # dλ_prev[N+1:end] .= dλ[N+1:end] .+ ( ( reshape( reshape(permutedims(m.hessianxp, [1,3,2]), N*M, N) * dx, N, M ) .+ reshape( reshape(m.hessianpp, N*M, M) * a.dp, N, M ) )' * λ[1:N] .+ m.jacobianp' * dλ[1:N] ) .* a.dt # bug!
    dλ_prev[N+1:end] .= dλ[N+1:end] .+ ( ( reshape( reshape(permutedims(m.hessianxp, [1,3,2]), N*M, N) * dx, N, M ) .+ reshape( reshape(m.hessianpp, N*M, M) * a.dp, N, M ) )' * λ[1:N] .+ m.jacobianp' * dλ_prev[1:N] ) .* a.dt # bug fixed

    # println("dλ_prev_previous:")
    # println(dλ_prev)
    #
    # dλ_prev[1] = 1. / (1. + a.dt * a.p[1]) * (-a.dt * λ[1] * a.dp[1] + K_over_obs_variance[1] * (dx[1] - K_over_obs_variance[1] * 2./Nobs[1] * x_minus_mean_obs[1] * x_minus_mean_obs_times_dx[1]))
    # dλ_prev[2] = -a.dt * (λ[1] * dx[1] + x[1] * dλ[1])
    # dλ_prev[3] = 0.
    # dλ_prev[4] = a.dt * (exp(a.p[3]) * a.p[3] * λ[1] + exp(a.p[3]) * dλ[1])
    #
    # println("dλ_prev_current:")
    # println(dλ_prev)

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

@views function _covariance!(a::Adjoint{N,L,R,U,K,T}, m::Model{N,L,R,U,T}, x_minus_mean_obs, u_minus_mean_obs, x_minus_mean_obs_times_dx, hv) where {N,L,R,U,K,T<:AbstractFloat}
    neighboring!(a, m)
    for _j in 1:N
        x_minus_mean_obs_times_dx[_j] .= mapreduce(nanzero, +, m.d_observation.(a.x[_j,:]) .* x_minus_mean_obs[_j,:] .* a.dx[_j,:])
    end
    u_minus_mean_obs_times_du = mapreduce(nanzero, +, u_minus_mean_obs .* a.dp[2:end])

    hessian_vector_product!(a, m, x_minus_mean_obs, u_minus_mean_obs, x_minus_mean_obs_times_dx, u_minus_mean_obs_times_du, hv)
end

@views function covariance!(a::Adjoint{N,L,R,U,K,T}, m::Model{N,L,R,U,T}) where {N,L,R,U,K,T<:AbstractFloat}
    fill!(a.dx[:,1], 0.)
    fill!(a.dp, 0.)
    hessian = Matrix{T}(L,L)

    x_minus_mean_obs = m.observation.(a.x) .- a.obs_mean[1:N,:] # bug fixed
    u_minus_mean_obs = (exp.(a.p[2:end]) .- a.obs_mean[N+1,:][a.finite[N+1,:]]) .* exp.(a.p[2:end])

    x_minus_mean_obs_times_dx = Vector{T}(N)

    # h11 = K/a.obs_variance[1] * (1. + 1./(1. + a.dt * a.p[1])^2. - ( a.x[1,1] - a.obs_mean[1,1] + (a.x[1,2] - a.obs_mean[1,2]) / (1. + a.dt * a.p[1]) )^2. / a.obs_variance[1] )
    # h22 = K/a.obs_variance[1] * (a.dt / (1. + a.dt * a.p[1]))^2. * a.x[1,2] * (3a.x[1,2] - 2a.obs_mean[1,2] - a.x[1,2] * x_minus_mean_obs[1,2]^2. / a.obs_variance[1]) + 1.
    # h33 = K/a.obs_variance[2] * exp(a.p[2]) * (2exp(a.p[2]) - a.obs_mean[2,1] - exp(a.p[2]) * (exp(a.p[2]) - a.obs_mean[2,1])^2. / a.obs_variance[2])
    # h44 = K/a.obs_variance[2] * exp(a.p[2]) * (2exp(a.p[2]) - a.obs_mean[2,1] - exp(a.p[2]) * (exp(a.p[2]) - a.obs_mean[2,1])^2. / a.obs_variance[2]) + K/a.obs_variance[1] * a.dt / (1. + a.dt*a.p[1]) * exp(a.p[3]) * ( a.x[1,2] - a.obs_mean[1,2] + a.dt / (1. + a.dt*a.p[1]) * exp(a.p[3]) * (1. - (a.x[1,2] - a.obs_mean[1,2])^2. / a.obs_variance[1]) )

    # h12 = -a.dt/(1.+a.dt*a.p[1]) * ( K/a.obs_variance[1] * x_minus_mean_obs[1,2] + a.x[2] * ( 1./(1.+a.dt*a.p[1]) * K/a.obs_variance[1] * (1. - x_minus_mean_obs[2]^2/a.obs_variance[1]) - K/a.obs_variance[1]^2 * x_minus_mean_obs[1] * x_minus_mean_obs[2] ) )
    # h14 = (1./(1.+a.dt*a.p[1]) * K/a.obs_variance[1] * (1. - x_minus_mean_obs[1,2]^2/a.obs_variance[1]) - K/a.obs_variance[1]^2 * x_minus_mean_obs[1,1] * x_minus_mean_obs[1,2]) * a.dt / (1. + a.dt * a.p[1]) * exp(a.p[3])

    for i in 1:N
        a.dx[i,1] = 1.
        _covariance!(a, m, x_minus_mean_obs, u_minus_mean_obs, x_minus_mean_obs_times_dx, hessian[:,i])
        a.dx[i,1] = 0.
    end
    for i in N+1:L
        a.dp[i-N] = 1.
        _covariance!(a, m, x_minus_mean_obs, u_minus_mean_obs, x_minus_mean_obs_times_dx, hessian[:,i])
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
