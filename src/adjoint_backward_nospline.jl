innovation_λ(m::Model, x, obs_mean, obs_variance_over_K, finite) = finite ? m.d_observation(x) * (m.observation(x) - obs_mean)/obs_variance_over_K : zero(x)

@views innovation_dλ(m::Model, x_i, dx_i, Nobs, K_over_obs_variance, x_minus_mean_obs_i, x_minus_mean_obs_times_dx, finite_i) = finite_i ? K_over_obs_variance * ( ( m.dd_observation(x_i) * x_minus_mean_obs_i + m.d_observation(x_i)^2 ) * dx_i - K_over_obs_variance * oftype(K_over_obs_variance, 2.) / Nobs * m.d_observation(x_i) * x_minus_mean_obs_i * x_minus_mean_obs_times_dx) : zero(K_over_obs_variance)


@views function orbit!(a::Adjoint{N,L,K,T}, m) where {N,L,K,T<:AbstractFloat}
    # res = zeros(N)
    # x_min = zeros(N)
    # res_min = zeros(N)
    for _i in 1:a.steps
        next_x!(a, m, a.dt*_i, a.x[:,_i], a.x[:,_i+1]) # Use next step's t
    end
    obs_variance!(a, m)
    nothing
end

@views function neighboring!(a, m)
    for _i in 1:a.steps
        next_dx!(a, m, a.dt*(_i-1), a.x[:,_i],            a.dx[:,_i], a.dx[:,_i+1])
    end
    nothing
end

@views function obs_variance!(a::Adjoint{N,L,K}, m::Model{N,L}) where {N,L,K}
    for _i in 1:N
        a.obs_variance[_i] = (a.pseudo_obs_TSS[_i] + K * (mapreduce(abs2, +, (m.observation.(a.x[_i,:]) .- a.obs_mean[_i,:])[a.finite[_i,:]]) + a.obs_filterd_var[_i])) / a.Nobs[_i]
    end
    nothing
end

@views function gradient!(a::Adjoint{N,L,K}, m::Model{N,L}, gr) where {N,L,K} # assuming x[:,1] .= x0; p .= p; orbit!(dxdt, t, x, p, dt); is already run. λ[:,1] is the gradient.
    obs_variance_over_K = a.obs_variance ./ K
    prev_λ!(a, m, a.t[end], a.x[:,end], zeros(L), a.λ[:,end], a.obs_mean[:,end], obs_variance_over_K, a.finite[:,end])
    for _i in a.steps:-1:2
        prev_λ!(a, m, a.t[_i], a.x[:,_i], a.λ[:,_i+1], a.λ[:,_i], a.obs_mean[:,_i], obs_variance_over_K, a.finite[:,_i])
    end
    gr[1:N]   .= a.λ[1:N,2]   .+ innovation_λ.(m, a.x[:,1], a.obs_mean[:,1], obs_variance_over_K, a.finite[:,1])
    gr[N+1:L] .= a.λ[N+1:L,2] .+ a.regularization .* a.p
    nothing
end

@views function hessian_vector_product!(a::Adjoint{N,L,K}, m::Model{N,L}, x_minus_mean_obs, x_minus_mean_obs_times_dx, hv) where {N,L,K} # assuming dx[:,1] .= dx0; dp .= dp; neighboring!(jacobian, dt, t, x, p, dx, dp); is already run. dλ[:,1] is the hessian_vector_product.
    K_over_obs_variance = K ./ a.obs_variance
    prev_dλ!(a, m, a.t[end], a.x[:,end], a.dx[:,end], a.λ[:,end], zeros(L), a.dλ[:,end], a.Nobs, K_over_obs_variance, x_minus_mean_obs[:,end], x_minus_mean_obs_times_dx, a.finite[:,end])
    for _i in a.steps:-1:2
        prev_dλ!(a, m, a.t[_i], a.x[:,_i], a.dx[:,_i], a.λ[:,_i], a.dλ[:,_i+1], a.dλ[:,_i], a.Nobs, K_over_obs_variance, x_minus_mean_obs[:,_i], x_minus_mean_obs_times_dx, a.finite[:,_i])
    end
    hv[1:N]   .= a.dλ[1:N,2] .+ innovation_dλ.(m, a.x[:,1], a.dx[:,1], a.Nobs, K_over_obs_variance, x_minus_mean_obs[:,1], x_minus_mean_obs_times_dx, a.finite[:,1])
    hv[N+1:L] .= a.dλ[N+1:L,2] .+ a.regularization .* a.dp
    nothing
end

@views function cost(a::Adjoint{N,L,K}) where {N,L,K} # assuming x[:,1] .= x0; orbit!(dxdt, t, x, p, dt); is already run
    c = a.regularization * dot(a.p, a.p)
    for _i in 1:N
        if a.Nobs[_i] > 0
            c += a.Nobs[_i] * log(a.obs_variance[_i])
        end
    end
    c / oftype(a.dt, 2.)
end

@views function negative_log_likelihood(a::Adjoint{N,L,K}, pseudo_obs, pseudo_obs_var) where {N,L,K} # bug! fix me!
    c = zero(a.dt)
    for _i in 1:N
        if a.Nobs[_i] > 0
            pseudo_obs_i_over_2_minus_1 = pseudo_obs[_i]/2. - 1.
            c += pseudo_obs_i_over_2_minus_1 * log(pseudo_obs[_i] * pseudo_obs_var[_i] / 2.) + lgamma(pseudo_obs_i_over_2_minus_1) + (a.Nobs[_i]-pseudo_obs[_i])/.2 * log(2.*pi) + a.Nobs[_i]/.2 * (1. + log(a.obs_variance[_i]))
        end
    end
    c
end

@views function initialize!(a::Adjoint{N}, θ) where {N}
    copy!(a.x[:,1], θ[1:N])
    copy!(a.p, θ[N+1:end])
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

@views function next_x!(a::Adjoint{N,L,K,T},       m::Model,       t, x_prev, x) where {N,L,K,T}
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

@views function prev_λ!( a::Adjoint{N},    m::Model{N},    t, x,                   λ, λ_prev, obs_mean, obs_variance_over_K, finite) where {N}
    inv_j!(a, m, t, x)
    m.jacobianp!(m, t, x, a.p)
    λ_prev[1:N] .= a.jacobian_inv' * (λ[1:N] .+ innovation_λ.(m, x, obs_mean, obs_variance_over_K, finite))
    λ_prev[N+1:end] .= λ[N+1:end] + m.jacobianp' * λ_prev[1:N] .* a.dt
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
    dλ_prev[N+1:end] .= dλ[N+1:end] .+ ( ( reshape( reshape(permutedims(m.hessianxp, [1,3,2]), N*M, N) * dx, N, M ) .+ reshape( reshape(m.hessianpp, N*M, M) * a.dp, N, M ) )' * λ[1:N] .+ m.jacobianp' * dλ[1:N] ) .* a.dt
    nothing
end

@views function fg!(F, ∇θ, θ, a::Adjoint{N}, m::Model{N}) where {N}
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

@views function minimize!(initial_θ, a::Adjoint{N,L,K,T}, m::Model{N,L,T}) where {N,L,K,T<:AbstractFloat} # Fixed. views is definitely needed for copy!
    initialize!(a, initial_θ)
    orbit!(a, m)
    F = cost(a)

    ∇θ = Vector{T}(L)
    gradient!(a, m, ∇θ)

    df = OnceDifferentiable(θ -> fg!(F, nothing, θ, a, m), (∇θ, θ) -> fg!(nothing, ∇θ, θ, a, m), (∇θ, θ) -> fg!(F, ∇θ, θ, a, m), initial_θ, F, ∇θ, inplace=true)
    options = Optim.Options(;x_tol=1e-32, f_tol=1e-32, g_tol=1e-8, iterations=1_000, store_trace=true, show_trace=false, show_every=1)
    # options = Optim.Options(;x_tol=1e-32, f_tol=-1., g_tol=-1., iterations=1_000, store_trace=true, show_trace=false, show_every=1)
    lbfgs_ls_scaled_hz = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true), linesearch = LineSearches.HagerZhang())
    # lbfgs_ls_alpha_hz = LBFGS(;alphaguess = LineSearches.InitialStatic(;alpha=0.001), linesearch = LineSearches.HagerZhang())
    # optimize(df, initial_p, lbfgs_ls_alpha_hz, options)
    optimize(df, initial_θ, lbfgs_ls_scaled_hz, options)
end

nanzero(x) = isnan(x) ? zero(x) : x

@views function _covariance!(a::Adjoint{N,L,K,T}, m::Model{N,L,T}, x_minus_mean_obs, x_minus_mean_obs_times_dx, hv) where {N,L,K,T<:AbstractFloat}
    neighboring!(a, m)
    for _j in 1:N
        x_minus_mean_obs_times_dx[_j] .= mapreduce(nanzero, +, m.d_observation.(a.x[_j,:]) .* x_minus_mean_obs[_j,:] .* a.dx[_j,:])
    end
    hessian_vector_product!(a, m, x_minus_mean_obs, x_minus_mean_obs_times_dx, hv)
end

@views function covariance!(a::Adjoint{N,L,K,T}, m::Model{N,L,T}) where {N,L,K,T<:AbstractFloat}
    fill!(a.dx[:,1], 0.)
    fill!(a.dp, 0.)
    hessian = Matrix{T}(L,L)

    x_minus_mean_obs = m.observation.(a.x) .- a.obs_mean
    x_minus_mean_obs_times_dx = Vector{T}(N)

    for i in 1:N
        a.dx[i,1] = 1.
        _covariance!(a, m, x_minus_mean_obs, x_minus_mean_obs_times_dx, hessian[:,i])
        a.dx[i,1] = 0.
    end
    for i in N+1:L
        a.dp[i-N] = 1.
        _covariance!(a, m, x_minus_mean_obs, x_minus_mean_obs_times_dx, hessian[:,i])
        a.dp[i-N] = 0.
    end
    θ = vcat(a.x[:,1], a.p)

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
        println("hessian")
        println(hessian)
        println("covariance:")
        println(covariance)
        if (minimum(diag(covariance)) < 0)
            println(STDERR, "Reason: Negative variance!")
        else
            println(STDERR, "Reason: Taking sqrt of variance failed due to $message")
        end
        return AssimilationResults(θ, a.obs_variance, hessian, covariance)
    end
    return AssimilationResults(θ, a.obs_variance, stddev, hessian, covariance)
end
