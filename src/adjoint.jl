innovation_λ(  x, obs, obs_variance) = isnan(obs) ? zero(x)  : (x - obs)/obs_variance # element-wise innovation of λ. fix me to be type invariant!
# innovation_λ(  x::T, obs::T, obs_variance::T) where {T <: AbstractFloat} = (x - obs)/obs_variance
# innovation_λ(  x::T, obs::NAtype, obs_variance::T) where {T <: AbstractFloat} = zero(x)

innovation_dλ(dx, obs, obs_variance) = isnan(obs) ? zero(dx) : dx/obs_variance        # element-wise innovation of dλ.


next_x!(        a::Adjoint,       m::Model,       t, x, x_nxt)                                               = (m.dxdt!(    m, t, x, a.p);                                     x_nxt .=  x .+ m.dxdt                                                             .* a.dt; nothing)

next_dx!(       a::Adjoint,       m::Model,       t, x,        dx, dx_nxt)                                   = (m.jacobian!(m, t, x, a.p);                                    dx_nxt .= dx .+ m.jacobian  * CatView(dx, a.dp)                                    .* a.dt; nothing)

@views prev_λ!( a::Adjoint{N},    m::Model{N},    t, x,                   λ, λ_prev)            where {N}    = (m.jacobian!(m, t, x, a.p);                                    λ_prev .=  λ .+ m.jacobian' *  λ[1:N]                                              .* a.dt; nothing)

@views prev_dλ!(a::Adjoint{N, L}, m::Model{N, L}, t, x,        dx,        λ,       dλ, dλ_prev) where {N, L} = (m.hessian!( m, t, x, a.p); prev_λ!(a, m, t, x, dλ, dλ_prev); dλ_prev .+= reshape(reshape(m.hessian, N*L, L) * CatView(dx, a.dp), N, L)' * λ[1:N] .* a.dt; nothing)


@views function orbit!(a, m)
    for _i in 1:a.steps
        next_x!(a, m, a.dt*(_i-1), a.x[:,_i], a.x[:,_i+1])
    end
    nothing
end

@views function neighboring!(a, m)
    for _i in 1:a.steps
        next_dx!(a, m, a.dt*(_i-1), a.x[:,_i],            a.dx[:,_i], a.dx[:,_i+1])
    end
    nothing
end

function gradient!(a::Adjoint{N,L,K}, m::Model{N,L}) where {N,L,K} # assuming x[:,1] .= x0; p .= p; orbit!(dxdt, t, x, p, dt); is already run. λ[:,1] is the gradient.
    a.λ[1:N,end] .= innovation_λ.(view(a.x, :, a.steps+1), view(a.obs, :, a.steps+1, 1), a.obs_variance)
    for _replicate in 2:K
        a.λ[1:N,end] .+= innovation_λ.(view(a.x, :, a.steps+1), view(a.obs, :, a.steps+1, _replicate), a.obs_variance)
    end
    for _i in a.steps:-1:1
        prev_λ!(a, m, a.dt*(_i-1), view(a.x, :, _i),                                      view(a.λ, :, _i+1), view(a.λ, :, _i)) # fix me! a.dt*_i?
        for _replicate in 1:K
            a.λ[1:N,_i] .+= innovation_λ.(view(a.x, :, _i), view(a.obs, :, _i, _replicate), a.obs_variance)
        end
    end
    nothing
end

function hessian_vector_product!(a::Adjoint{N,L,K}, m::Model{N,L}) where {N,L,K} # assuming dx[:,1] .= dx0; dp .= dp; neighboring!(jacobian, dt, t, x, p, dx, dp); is already run. dλ[:,1] is the hessian_vector_product.
    a.dλ[1:N,end] .= innovation_dλ.(view(a.dx, :, a.steps+1), view(a.obs, :, a.steps+1, 1), a.obs_variance)
    for _replicate in 2:K
        a.dλ[1:N,end] .+= innovation_dλ.(view(a.dx, :, a.steps+1), view(a.obs, :, a.steps+1, _replicate), a.obs_variance)
    end
    for _i in a.steps:-1:1
        prev_dλ!(a, m, a.dt*(_i-1), view(a.x, :, _i),            view(a.dx, :, _i),              view(a.λ, :, _i+1),            view(a.dλ, :, _i+1), view(a.dλ, :, _i)) # fix me! a.dt*_i?
        for _replicate in 1:K
            a.dλ[1:N,_i] .+= innovation_dλ.(view(a.dx, :, _i), view(a.obs, :, _i, _replicate), a.obs_variance)
        end
    end
    nothing
end


@views function cost(a::Adjoint{N,L,K}) where {N,L,K} # assuming x[:,1] .= x0; orbit!(dxdt, t, x, p, dt); is already run
    c = zero(a.obs_variance)
    for _replicate in 1:K
        c += mapreduce(abs2, +, (a.x .- a.obs[:,:,_replicate])[isfinite.(a.obs[:,:,_replicate])])
    end
    c / a.obs_variance / oftype(a.obs_variance, 2.)
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
        gradient!(a, m)
        copy!(∇θ, a.λ[:,1])
        nothing
    end
    if !(F == nothing)
        F = cost(a)
    end
end

# function minimize!(initial_θ, a)
#     df = NLSolversBase.OnceDifferentiable(NLSolversBase.only_fg!((F, ∇θ, θ) -> fg!(F, ∇θ, θ, a)), initial_θ)
#     Optim.optimize(df, initial_θ, LBFGS())
# end

@views function minimize!(initial_θ, a::Adjoint{N,L}, m::Model{N,L}) where {N,L} # Fixed. views is definitely needed for copy!
    initialize!(a, initial_θ)
    orbit!(a, m)
    F = cost(a)

    gradient!(a, m)
    ∇θ = zeros(L)
    copy!(∇θ, a.λ[:,1])

    df = OnceDifferentiable(θ -> fg!(F, nothing, θ, a, m), (∇θ, θ) -> fg!(nothing, ∇θ, θ, a, m), (∇θ, θ) -> fg!(F, ∇θ, θ, a, m), initial_θ, F, ∇θ, inplace=true)
    options = Optim.Options(;x_tol=1e-32, f_tol=1e-32, g_tol=1e-8, iterations=1_000, store_trace=true, show_trace=false, show_every=1)
    optimize(df, initial_θ, LBFGS(), options)
end

@views function covariance!(a::Adjoint{N,L,K,T}, m::Model{N,L,T}) where {N,L,K,T<:AbstractFloat}
    fill!(a.dx[:,1], 0.)
    fill!(a.dp, 0.)
    hessian = Matrix{T}(L,L)
    for i in 1:N
        a.dx[i,1] = 1.
        neighboring!(a, m)
        hessian_vector_product!(a, m)
        copy!(hessian[:,i], a.dλ[:,1])
        a.dx[i,1] = 0.
    end
    for i in N+1:L
        a.dp[i-N] = 1.
        neighboring!(a, m)
        hessian_vector_product!(a, m)
        copy!(hessian[:,i], a.dλ[:,1])
        a.dp[i-N] = 0.
    end
    θ = vcat(a.x[:,1], a.p)
    covariance = nothing
    try
        covariance = inv(hessian)
    catch message
        println(STDERR, "CI calculation failed.\nReason: Hessian inversion failed due to $message")
        return AssimilationResults(θ)
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
        return AssimilationResults(θ, covariance)
    end
    return AssimilationResults(θ, stddev, covariance)
end
