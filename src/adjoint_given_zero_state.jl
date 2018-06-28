innovation_λ(m::Model, x, obs_mean, obs_variance_over_K, finite) = finite ? m.d_observation(x) * (m.observation(x) - obs_mean)/obs_variance_over_K : zero(x)

@views innovation_dλ(m::Model, x_i, dx_i, Nobs, K_over_obs_variance, x_minus_mean_obs_i, x_minus_mean_obs_times_dx, finite_i) = finite_i ? K_over_obs_variance * ( ( m.dd_observation(x_i) * x_minus_mean_obs_i + m.d_observation(x_i)^2 ) * dx_i - K_over_obs_variance * oftype(K_over_obs_variance, 2.) / Nobs * m.d_observation(x_i) * x_minus_mean_obs_i * x_minus_mean_obs_times_dx) : zero(K_over_obs_variance)


next_x!(        a::Adjoint,       m::Model,       t, x, x_nxt)                                               = (m.dxdt!(    m, t, x, a.p);                                     x_nxt .=  x .+ m.dxdt                                                             .* a.dt; nothing)

next_dx!(       a::Adjoint,       m::Model,       t, x,        dx, dx_nxt)                                   = (m.jacobian!(m, t, x, a.p);                                    dx_nxt .= dx .+ m.jacobian  * CatView(dx, a.dp)                                    .* a.dt; nothing)

@views prev_λ!( a::Adjoint{N},    m::Model{N},    t, x,                   λ, λ_prev)            where {N}    = (m.jacobian!(m, t, x, a.p);                                    λ_prev .=  λ .+ m.jacobian' *  λ[1:N]                                              .* a.dt; nothing)

@views prev_dλ!(a::Adjoint{N, L}, m::Model{N, L}, t, x,        dx,        λ,       dλ, dλ_prev) where {N, L} = (m.hessian!( m, t, x, a.p); prev_λ!(a, m, t, x, dλ, dλ_prev); dλ_prev .+= reshape(reshape(m.hessian, N*L, L) * CatView(dx, a.dp), N, L)' * λ[1:N] .* a.dt; nothing)


@views function orbit!(a, m)
    for _i in 1:a.steps
        next_x!(a, m, a.dt*(_i-1), a.x[:,_i], a.x[:,_i+1])
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
    a.λ[1:N,end] .= innovation_λ.(m, a.x[:,a.steps+1], a.obs_mean[:,a.steps+1], obs_variance_over_K, a.finite[:,a.steps+1])
    for _i in a.steps:-1:1
        prev_λ!(a, m, a.dt*(_i-1),　a.x[:,_i],            a.λ[:,_i+1], a.λ[:,_i])
        a.λ[1:N,_i] .+= innovation_λ.(m, a.x[:,_i], a.obs_mean[:,_i], obs_variance_over_K, a.finite[:,_i])
    end
    gr .= a.λ[N+1:L,1]
    nothing
end

@views function hessian_vector_product!(a::Adjoint{N,L,K}, m::Model{N,L}, x_minus_mean_obs, x_minus_mean_obs_times_dx, hv) where {N,L,K} # assuming dx[:,1] .= dx0; dp .= dp; neighboring!(jacobian, dt, t, x, p, dx, dp); is already run. dλ[:,1] is the hessian_vector_product.
    K_over_obs_variance = K ./ a.obs_variance
    a.dλ[1:N, a.steps+1] .= innovation_dλ.(m, a.x[:,a.steps+1], a.dx[:,a.steps+1], a.Nobs, K_over_obs_variance, x_minus_mean_obs[:,a.steps+1], x_minus_mean_obs_times_dx, a.finite[:,a.steps+1])
    for _i in a.steps:-1:1
        prev_dλ!(a, m, a.dt*(_i-1), a.x[:,_i], a.dx[:,_i], a.λ[:,_i+1],            a.dλ[:,_i+1], a.dλ[:,_i])
        a.dλ[1:N,_i] .+= innovation_dλ.(m, a.x[:,_i], a.dx[:,_i], a.Nobs, K_over_obs_variance, x_minus_mean_obs[:,_i], x_minus_mean_obs_times_dx, a.finite[:,_i])
    end
    hv .= a.dλ[N+1:L,1]
    nothing
end

@views function cost(a::Adjoint{N,L,K}) where {N,L,K} # assuming x[:,1] .= x0; orbit!(dxdt, t, x, p, dt); is already run
    c = zero(a.dt)
    for _i in 1:N
        if a.Nobs[_i] > 0
            c += a.Nobs[_i] * log(a.obs_variance[_i])
        end
    end
    c / oftype(a.dt, 2.)
end

# @views function initialize!(a::Adjoint{N}, θ) where {N}
#     copy!(a.x[:,1], θ[1:N])
#     copy!(a.p, θ[N+1:end])
#     nothing
# end

@views function initialize_p!(a::Adjoint{N}, p) where {N}
    copy!(a.p, p)
    nothing
end

@views function fg!(F, ∇p, p, a::Adjoint{N}, m::Model{N}) where {N}
    initialize_p!(a, p)
    orbit!(a, m)
    if !(∇p == nothing)
        gradient!(a, m, ∇p)
    end
    if !(F == nothing)
        F = cost(a)
    end
end

# function minimize!(initial_θ, a, m)
#     df = NLSolversBase.OnceDifferentiable(NLSolversBase.only_fg!((F, ∇θ, θ) -> fg!(F, ∇θ, θ, a, m)), initial_θ)
#     Optim.optimize(df, initial_θ, LBFGS())
# end

@views function minimize!(initial_p, a::Adjoint{N,L,K,T}, m::Model{N,L,T}) where {N,L,K,T<:AbstractFloat} # Fixed. views is definitely needed for copy!
    initialize_p!(a, initial_p)
    orbit!(a, m)
    F = cost(a)

    ∇p = Vector{T}(L-N)
    gradient!(a, m, ∇p)

    df = OnceDifferentiable(p -> fg!(F, nothing, p, a, m), (∇p, p) -> fg!(nothing, ∇p, p, a, m), (∇p, p) -> fg!(F, ∇p, p, a, m), initial_p, F, ∇p, inplace=true)
    options = Optim.Options(;x_tol=1e-32, f_tol=1e-32, g_tol=1e-8, iterations=1_000, store_trace=true, show_trace=false, show_every=1)
    optimize(df, initial_p, LBFGS(), options)
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
    hessian = Matrix{T}(L-N,L-N)

    x_minus_mean_obs = m.observation.(a.x) .- a.obs_mean
    x_minus_mean_obs_times_dx = Vector{T}(N)

    for i in 1:L-N
        a.dp[i] = 1.
        _covariance!(a, m, x_minus_mean_obs, x_minus_mean_obs_times_dx, hessian[:,i])
        a.dp[i] = 0.
    end
    covariance = nothing
    try
        covariance = inv(hessian)
    catch message
        println(STDERR, "CI calculation failed.\nReason: Hessian inversion failed due to $message")
        return AssimilationResults(a.p, a.obs_variance)
    end
    stddev = nothing
    try
        stddev = sqrt.(diag(covariance))
    catch message
        if (minimum(diag(covariance)) < 0)
            println(STDERR, "CI calculation failed.\nReason: Negative variance!")
        else
            println(STDERR, "CI calculation failed.\nReason: Taking sqrt of variance failed due to $message")
        end
        return AssimilationResults(a.p, a.obs_variance, covariance)
    end
    return AssimilationResults(a.p, a.obs_variance, stddev, covariance)
end
