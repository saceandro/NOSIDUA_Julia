using Juno

innovation_λ(  x, obs, obs_variance) = isnan(obs) ? zero(x)  : (x - obs)/obs_variance # element-wise innovation of λ. fix me to be type invariant!
# innovation_λ(  x::T, obs::T, obs_variance::T) where {T <: AbstractFloat} = (x - obs)/obs_variance
# innovation_λ(  x::T, obs::NAtype, obs_variance::T) where {T <: AbstractFloat} = zero(x)

# innovation_dλ(dx, obs, obs_variance) = isnan(obs) ? zero(dx) : dx/obs_variance        # element-wise innovation of dλ.

# @views innovation_dλ(a::Adjoint{N,L,K}, i) where {N,L,K} = [any(isnan.(a.obs[k,i,:])) ? zero(a.dt) : K / a.obs_variance[k] * (a.dx[k,i] - K / a.obs_variance[k] * 2. / count(isfinite.(a.obs[k,:,:])) * (a.x[k,i] - mean(a.obs[k,i,:])) * dot( (a.x[k,:] .- [mean(a.obs[k,_i,:]) for _i in 1:a.steps+1] )[[ all(isfinite.(a.obs[1,_j,:])) for _j in 1:a.steps+1 ]] , a.dx[k,:][[all(isfinite.(a.obs[1,_j,:])) for _j in 1:a.steps+1 ]]) ) for k in 1:N]
@views innovation_dλ(a::Adjoint{N,L,K}, i) where {N,L,K} = [any(isnan.(a.obs[k,i,:])) ? zero(a.dt) : 1. / a.obs_variance[k] * (a.dx[k,i] - 1. / a.obs_variance[k] * 2. / count(isfinite.(a.obs[k,:,:])) * (a.x[k,i] - mean(a.obs[k,i,:])) * dot( (a.x[k,:] .- [mean(a.obs[k,_i,:]) for _i in 1:a.steps+1] )[[ all(isfinite.(a.obs[1,_j,:])) for _j in 1:a.steps+1 ]] , a.dx[k,:][[all(isfinite.(a.obs[1,_j,:])) for _j in 1:a.steps+1 ]]) ) for k in 1:N]


next_x!(        a::Adjoint,       m::Model,       i0, i)                                    = (m.dxdt!(     m, i0, i, a.t, a.x, a.p);                                                                        a.x[:,i+1] .= a.x[:,i] .+ m.dxdt .* a.dt;                                                                                                        nothing)

@views next_dx!(a::Adjoint,       m::Model,       i0, i)                                    = (m.jacobian!( m, i0, i, a.t, a.x, a.p); m.jacobian0!(m, i0, i, a.t, a.x, a.p);                                a.dx[:,i+1] .= a.dx[:,i] .+ ( m.jacobian * CatView(a.dx[:,i], a.dp) + m.jacobian0 * a.dx[:,1] ) .* a.dt;                                          nothing)

@views prev_λ!( a::Adjoint{N},    m::Model{N},    i0, i, λ, λ_prev)            where {N}    = (m.jacobian!( m, i0, i, a.t, a.x, a.p);                                                                            λ_prev .= λ .+ m.jacobian' *  λ[1:N] .* a.dt;                                                                                                nothing)

@views prev_dλ!(a::Adjoint{N, L}, m::Model{N, L}, i0, i, λ,       dλ, dλ_prev) where {N, L} = (m.hessian!(  m, i0, i, a.t, a.x, a.p); m.hessian0!( m, i0, i, a.t, a.x, a.p); prev_λ!(a, m, i0, i, dλ, dλ_prev); dλ_prev .+= reshape(reshape(m.hessian, N*L, L) * CatView(a.dx[:,i], a.dp) + reshape(m.hessian0, N*L, N) * a.dx[:,1], N, L)' * λ[1:N] .* a.dt; nothing)

@views jacobian0_λ!(a::Adjoint{N}, m::Model{N},   i0, i, acc)                  where {N}    = (m.jacobian0!(m, i0, i, a.t, a.x, a.p);                                        acc .+= m.jacobian0' * a.λ[1:N,i+1]                                                 .* a.dt; nothing)

# @views residual_dλ!(a::Adjoint{N,L}, m::Model{N,L},   i0, i, λ, dλ, acc)       where {N,L}  = (m.jacobian0!(m, i0, i, a.t, a.x, a.p); m.hessian0!( m, i0, i, a.t, a.x, a.p); m.hessian00!( m, i0, i, a.t, a.x, a.p); acc .+= ( reshape( reshape(m.hessian00, N*N, N) * a.dx[:,1] +  reshape( reshape(m.hessian0, N*L, N)', N*N, L ) * CatView(a.dx[:,i], a.dp), N, N)' * λ[1:N] + m.jacobian0' * dλ[1:N] ) .* a.dt; nothing)

__residual_dλ!(a::Adjoint,        m::Model,       i0, i)                                    = (m.jacobian0!(m, i0, i, a.t, a.x, a.p); m.hessian0!( m, i0, i, a.t, a.x, a.p); m.hessian00!( m, i0, i, a.t, a.x, a.p))
@views _residual_dλ!(a::Adjoint{N,L}, m::Model{N,L},  i0, i, λ, dλ, acc)       where {N,L}  = (acc .+= ( reshape( reshape(m.hessian00, N*N, N) * a.dx[:,1] +  reshape( permutedims(m.hessian0, [1,3,2]), N*N, L ) * CatView(a.dx[:,i], a.dp), N, N)' * λ[1:N] + m.jacobian0' * dλ[1:N] ) .* a.dt; nothing)
residual_dλ!(a::Adjoint{N,L},     m::Model{N, L}, i0, i, λ, dλ, acc)           where {N,L}  = (__residual_dλ!(a, m, i0, i); _residual_dλ!(a, m, i0, i, λ, dλ, acc))

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
    fill!(a.obs_variance, 0.)
    for _replicate in 1:K
        for _i in 1:N
            a.obs_variance[_i] += mapreduce(abs2, +, (a.x[_i,:] .- a.obs[_i,:,_replicate])[isfinite.(a.obs[_i,:,_replicate])])
        end
        # println(STDERR, "obsvar unnormalized:\t", a.obs_variance)
    end
    for _i in 1:N
        a.obs_variance[_i] /= count(isfinite.(a.obs[_i,:,:]))
    end
    # println(STDERR, "notnan count:\t", count(isfinite.(a.obs[1,:,:])), "\t", count(isfinite.(a.obs[2,:,:])))
    # println(STDERR, "obsvar:\t", a.obs_variance)
    nothing
end

function gradient!(a::Adjoint{N,L,K}, m::Model{N,L}) where {N,L,K} # assuming x[:,1] .= x0; p .= p; orbit!(dxdt, t, x, p, dt); is already run. λ[:,1] is the gradient.
    a.λ[1:N,end] .= innovation_λ.(view(a.x, :, a.steps+1), view(a.obs, :, a.steps+1, 1), a.obs_variance)
    for _replicate in 2:K
        a.λ[1:N,end] .+= innovation_λ.(view(a.x, :, a.steps+1), view(a.obs, :, a.steps+1, _replicate), a.obs_variance)
    end
    for _i in a.steps:-1:1
        prev_λ!(a, m, 1, _i, view(a.λ, :, _i+1), view(a.λ, :, _i)) # fix me! a.dt*_i?
        for _replicate in 1:K
            a.λ[1:N,_i] .+= innovation_λ.(view(a.x, :, _i), view(a.obs, :, _i, _replicate), a.obs_variance)
        end
    end
    acc = zeros(N)
    for _i in a.steps:-1:1
        jacobian0_λ!(a, m, 1, _i, acc)
    end
    # println("acc:\t", acc, "\tgrad:\t", CatView(acc .+ a.λ[1:N,1], a.λ[N+1:L,1]))
    [acc .+ a.λ[1:N,1]; a.λ[N+1:L,1]]
end

function hessian_vector_product!(a::Adjoint{N,L,K}, m::Model{N,L}) where {N,L,K} # assuming dx[:,1] .= dx0; dp .= dp; neighboring!(jacobian, dt, t, x, p, dx, dp); is already run. dλ[:,1] is the hessian_vector_product.
    a.dλ[1:N,end] .= innovation_dλ(a, a.steps+1)
    for _i in a.steps:-1:1
        prev_dλ!(a, m, 1, _i, view(a.λ, :, _i+1), view(a.dλ, :, _i+1), view(a.dλ, :, _i)) # fix me! a.dt*_i?
        for _replicate in 1:K
            a.dλ[1:N,_i] .+= innovation_dλ(a, _i)
        end
    end
    acc = zeros(N)
    for _i in a.steps:-1:1
        residual_dλ!(a, m, 1, _i, view(a.λ, :, _i+1), view(a.dλ, :, _i+1), acc)
    end
    println("acc:\t", acc, "\tdλ[:,1]:\t", a.dλ[:,1])
    [acc .+ a.dλ[1:N,1]; a.dλ[N+1:L,1]]
end


# @views function cost(a::Adjoint{N,L,K}) where {N,L,K} # assuming x[:,1] .= x0; orbit!(dxdt, t, x, p, dt); is already run
#     c = zero(a.dt)
#     for _i in 1:N
#         observations = count(isfinite.(a.obs[_i,:,:]))
#         if observations > 0
#             c += observations * log(a.obs_variance[_i])
#         end
#     end
#     for _replicate in 1:K
#         for _i in 1:N
#             if count(isfinite.(a.obs[_i,:,_replicate])) > 0
#                 c += mapreduce(abs2, +, (a.x[_i,:] .- a.obs[_i,:,_replicate])[isfinite.(a.obs[_i,:,_replicate])]) / a.obs_variance[_i]
#             end
#         end
#     end
#     c / oftype(a.dt, 2.)
# end

@views function cost(a::Adjoint{N,L,K}) where {N,L,K} # assuming x[:,1] .= x0; orbit!(dxdt, t, x, p, dt); is already run
    c = zero(a.dt)
    for _i in 1:N
        observations = count(isfinite.(a.obs[_i,:,:]))
        if  observations > 0
            c += observations * log(a.obs_variance[_i])
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
        ∇θ .= gradient!(a, m)
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

    ∇θ = gradient!(a, m)

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
        hessian[:,i] .= hessian_vector_product!(a, m)
        a.dx[i,1] = 0.
    end
    for i in N+1:L
        a.dp[i-N] = 1.
        neighboring!(a, m)
        hessian[:,i] .= hessian_vector_product!(a, m)
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
