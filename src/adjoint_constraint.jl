using Juno

innovation_λ(  x, obs_mean, obs_variance, finite) = finite ? (x - obs_mean)/obs_variance : zero(x)

@views innovation_dλ(dx_i, Nobs, K_over_obs_variance, x_minus_mean_obs_i, x_minus_mean_obs_times_dx, finite_i) = finite_i ? K_over_obs_variance * (dx_i - K_over_obs_variance * oftype(K_over_obs_variance, 2.) / Nobs * x_minus_mean_obs_i * x_minus_mean_obs_times_dx) : zero(K_over_obs_variance)


next_x!(        a::Adjoint,       m::Model,       i0, i)                                    = (m.dxdt!(     m, i0, i, a.t, a.x, a.p);                                                                        a.x[:,i+1] .= a.x[:,i] .+ m.dxdt .* a.dt;                                                                                                        nothing)

@views next_dx!(a::Adjoint,       m::Model,       i0, i)                                    = (m.jacobian!( m, i0, i, a.t, a.x, a.p); m.jacobian0!(m, i0, i, a.t, a.x, a.p);                                a.dx[:,i+1] .= a.dx[:,i] .+ ( m.jacobian * CatView(a.dx[:,i], a.dp) + m.jacobian0 * a.dx[:,1] ) .* a.dt;                                          nothing)

@views prev_λ!( a::Adjoint{N},    m::Model{N},    i0, i, λ, λ_prev)            where {N}    = (m.jacobian!( m, i0, i, a.t, a.x, a.p);                                                                            λ_prev .= λ .+ m.jacobian' *  λ[1:N] .* a.dt;                                                                                                nothing)

@views prev_dλ!(a::Adjoint{N, L}, m::Model{N, L}, i0, i, λ,       dλ, dλ_prev) where {N, L} = (m.hessian!(  m, i0, i, a.t, a.x, a.p); m.hessian0!( m, i0, i, a.t, a.x, a.p); prev_λ!(a, m, i0, i, dλ, dλ_prev); dλ_prev .+= reshape(reshape(m.hessian, N*L, L) * CatView(a.dx[:,i], a.dp) + reshape(m.hessian0, N*L, N) * a.dx[:,1], N, L)' * λ[1:N] .* a.dt; nothing)

@views jacobian0_λ!(a::Adjoint{N}, m::Model{N},   i0, i, acc)                  where {N}    = (m.jacobian0!(m, i0, i, a.t, a.x, a.p);                                                                               acc .+= m.jacobian0' * a.λ[1:N,i+1]                                                                                              .* a.dt; nothing)

@views function residual_dλ!(a::Adjoint{N,L}, m::Model{N, L}, i0, i, λ, dλ, acc) where {N,L}
    m.jacobian0!(m, i0, i, a.t, a.x, a.p)
    m.hessian0!( m, i0, i, a.t, a.x, a.p)
    m.hessian00!( m, i0, i, a.t, a.x, a.p)
    acc .+= ( reshape( reshape(m.hessian00, N*N, N) * a.dx[:,1] + reshape( permutedims(m.hessian0, [1,3,2]), N*N, L ) * CatView(a.dx[:,i], a.dp), N, N)' * λ[1:N] + m.jacobian0' * dλ[1:N] ) .* a.dt
    nothing
end

@views function orbit!(a, m)
    for _i in 1:a.steps
        next_x!(a, m, 1, _i)
    end
    obs_variance!(a)
    nothing
end

@views function neighboring!(a, m)
    for _i in 1:a.steps
        next_dx!(a, m, 1, _i)
    end
    nothing
end

@views function obs_variance!(a::Adjoint{N,L,K}) where {N,L,K}
    for _i in 1:N
        a.obs_variance[_i] = (a.pseudo_obs_TSS[_i] + K * (mapreduce(abs2, +, (a.x[_i,:] .- a.obs_mean[_i,:])[a.finite[_i,:]]) + a.obs_filterd_var[_i])) / a.Nobs[_i]
    end
    nothing
end

@views function gradient!(a::Adjoint{N,L,K}, m::Model{N,L}, gr) where {N,L,K} # assuming x[:,1] .= x0; p .= p; orbit!(dxdt, t, x, p, dt); is already run. λ[:,1] is the gradient.
    obs_variance_over_K = a.obs_variance ./ K
    a.λ[1:N,end] .= innovation_λ.(a.x[:,a.steps+1], a.obs_mean[:,a.steps+1], obs_variance_over_K, a.finite[:,a.steps+1])
    for _i in a.steps:-1:1
        prev_λ!(a, m, 1, _i, a.λ[:,_i+1], a.λ[:,_i]) # fix me! a.dt*_i?
        a.λ[1:N,_i] .+= innovation_λ.(a.x[:,_i], a.obs_mean[:,_i], obs_variance_over_K, a.finite[:,_i])
    end
    gr .= a.λ[:,1]
    for _i in a.steps:-1:1
        jacobian0_λ!(a, m, 1, _i, gr[1:N])
    end
    nothing
end

@views function hessian_vector_product!(a::Adjoint{N,L,K}, m::Model{N,L}, x_minus_mean_obs, x_minus_mean_obs_times_dx, hv) where {N,L,K} # assuming dx[:,1] .= dx0; dp .= dp; neighboring!(jacobian, dt, t, x, p, dx, dp); is already run. dλ[:,1] is the hessian_vector_product.
    K_over_obs_variance = K ./ a.obs_variance
    a.dλ[1:N, a.steps+1] .= innovation_dλ.(a.dx[:,a.steps+1], a.Nobs, K_over_obs_variance, x_minus_mean_obs[:,a.steps+1], x_minus_mean_obs_times_dx, a.finite[:,a.steps+1])
    for _i in a.steps:-1:1
        prev_dλ!(a, m, 1, _i, a.λ[:,_i+1], a.dλ[:,_i+1], a.dλ[:,_i]) # fix me! a.dt*_i?
        a.dλ[1:N,_i] .+= innovation_dλ.(a.dx[:,_i], a.Nobs, K_over_obs_variance, x_minus_mean_obs[:,_i], x_minus_mean_obs_times_dx, a.finite[:,_i])
    end
    hv .= a.dλ[:,1]
    for _i in a.steps:-1:1
        residual_dλ!(a, m, 1, _i, a.λ[:,_i+1], a.dλ[:,_i+1], hv[1:N])
    end
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

@views function initialize!(a::Adjoint{N}, θ) where {N}
    copy!(a.x[:,1], θ[1:N])
    copy!(a.p, θ[N+1:end])
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
    optimize(df, initial_θ, LBFGS(), options)
end

@views function _covariance!(a::Adjoint{N,L,K,T}, m::Model{N,L,T}, x_minus_mean_obs, x_minus_mean_obs_times_dx, hv) where {N,L,K,T<:AbstractFloat}
    neighboring!(a, m)
    for _j in 1:N
        x_minus_mean_obs_times_dx[_j] .= dot(x_minus_mean_obs[_j,:][a.finite[_j,:]], a.dx[_j,:][a.finite[_j,:]])
    end
    hessian_vector_product!(a, m, x_minus_mean_obs, x_minus_mean_obs_times_dx, hv)
end

@views function covariance!(a::Adjoint{N,L,K,T}, m::Model{N,L,T}) where {N,L,K,T<:AbstractFloat}
    fill!(a.dx[:,1], 0.)
    fill!(a.dp, 0.)
    hessian = Matrix{T}(L,L)

    x_minus_mean_obs = a.x .- a.obs_mean
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
        return AssimilationResults(θ, a.obs_variance)
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
        return AssimilationResults(θ, a.obs_variance, covariance)
    end
    return AssimilationResults(θ, a.obs_variance, stddev, covariance)
end
