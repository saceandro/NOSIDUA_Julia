innovation_λ(  x, obs, obs_variance) = isnan(obs) ? zero(x)  : (x - obs)/obs_variance # element-wise innovation of λ. fix me to be type invariant!
# innovation_λ(  x::T, obs::T, obs_variance::T) where {T <: AbstractFloat} = (x - obs)/obs_variance
# innovation_λ(  x::T, obs::NAtype, obs_variance::T) where {T <: AbstractFloat} = zero(x)

innovation_dλ(dx, obs, obs_variance) = isnan(obs) ? zero(dx) : dx/obs_variance        # element-wise innovation of dλ.

next_x!(        a::Adjoint,       t, x, x_nxt)                                               = (a.model.dxdt!(    a.model.dxdt,     t, x, a.p);                                  x_nxt .=  x .+ a.model.dxdt                                                             .* a.dt; nothing)

next_dx!(       a::Adjoint,       t, x,        dx, dx_nxt)                                   = (a.model.jacobian!(a.model.jacobian, t, x, a.p);                                 dx_nxt .= dx .+ a.model.jacobian  * CatView(dx, a.dp)                                    .* a.dt; nothing)

@views prev_λ!( a::Adjoint{N},    t, x,                   λ, λ_prev)            where {N}    = (a.model.jacobian!(a.model.jacobian, t, x, a.p);                                 λ_prev .=  λ .+ a.model.jacobian' *  λ[1:N]                                              .* a.dt; nothing)

@views prev_dλ!(a::Adjoint{N, L}, t, x,        dx,        λ,       dλ, dλ_prev) where {N, L} = (a.model.hessian!( a.model.hessian,  t, x, a.p); prev_λ!(a, t, x, dλ, dλ_prev); dλ_prev .+= reshape(reshape(a.model.hessian, N*L, L) * CatView(dx, a.dp), N, L)' * λ[1:N] .* a.dt; nothing)


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
    for _i in a.steps:-1:1
        prev_λ!(a, a.dt*(_i-1), a.x[:,_i],                                      a.λ[:,_i+1], a.λ[:,_i]) # fix me! a.dt*_i?
        a.λ[1:N,_i] .+= innovation_λ.(a.x[:,_i], a.obs[:,_i], a.obs_variance)
    end
    nothing
end

orbit_gradient!(a) = (orbit!(a); gradient!(a); a.λ[:,1])

function numerical_gradient!(a::Adjoint{N,L}, gr, h) where {N,L}
    c = orbit_cost!(a)
    for _i in 1:N
        a.x[_i,1] += h
        gr[_i] = (orbit_cost!(a) - c)/h
        a.x[_i,1] -= h
    end
    for _i in N+1:L
        a.p[_i-N] += h
        gr[_i] = (orbit_cost!(a) - c)/h
        a.p[_i-N] -= h
    end
    nothing
end

@views function hessian_vector_product!(a::Adjoint{N}) where {N} # assuming dx[:,1] .= dx0; dp .= dp; neighboring!(jacobian, dt, t, x, p, dx, dp); is already run. dλ[:,1] is the hessian_vector_product.
    a.dλ[1:N,end] .= innovation_dλ.(a.dx[:,end], a.obs[:,end], a.obs_variance)
    for _i in a.steps:-1:1
        prev_dλ!(a, a.dt*(_i-1), a.x[:,_i],            a.dx[:,_i],              a.λ[:,_i+1],            a.dλ[:,_i+1], a.dλ[:,_i]) # fix me! a.dt*_i?
        a.dλ[1:N,_i] .+= innovation_dλ.(a.dx[:,_i], a.obs[:,_i], a.obs_variance)
    end
    nothing
end

@views cost(a) = mapreduce(abs2, +, (a.x .- a.obs)[isfinite.(a.obs)]) / a.obs_variance / oftype(a.obs_variance, 2.) # assuming x[:,1] .= x0; orbit!(dxdt, t, x, p, dt); is already run

orbit_cost!(a) = (orbit!(a); cost(a))

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

    gradient!(a)
    ∇θ = zeros(L)
    copy!(∇θ, a.λ[:,1])

    df = OnceDifferentiable(θ -> fg!(F, nothing, θ, a), (∇θ, θ) -> fg!(nothing, ∇θ, θ, a), (∇θ, θ) -> fg!(F, ∇θ, θ, a), initial_θ, F, ∇θ, inplace=true)
    # options = Options(;x_tol=1e-32, f_tol=1e-32, g_tol=1e-8, iterations=1_000, show_every=1)
    optimize(df, initial_θ, LBFGS())
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
    covariance .= inv(hessian)
    variance .= diag(covariance)
    stddev .= sqrt.(variance)
    nothing
end

function numerical_hessian!(a::Adjoint{N,L}, hessian, h) where {N,L}
    gr = orbit_gradient!(a)
    for _i in 1:N
        a.x[_i,1] += h
        hessian[:,_i] .= (orbit_gradient!(a) .- gr)/h
        a.x[_i,1] -= h
    end
    for _i in N+1:L
        a.p[_i-N] += h
        hessian[:,_i] .= (orbit_gradient!(a) .- gr)/h
        a.p[_i-N] -= h
    end
    nothing
end
