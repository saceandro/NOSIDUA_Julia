using Missings, DataArrays, CatViews, Optim

function dxdt!(dxdt::Vector{Float64}, t::Float64, x::Vector{Float64}, p::Vector{Float64})
    N = length(x)
    dxdt[1]     = p[2] * (x[2]   - x[N-1]) * x[N]   + p[1] - x[1]
    dxdt[2]     = p[2] * (x[3]   - x[N])   * x[1]   + p[1] - x[2]
    @simd for i in 3:N-1
        dxdt[i] = p[2] * (x[i+1] - x[i-2]) * x[i-1] + p[1] - x[i]
    end
    dxdt[N]     = p[2] * (x[1]   - x[N-2]) * x[N-1] + p[1] - x[N]
    dxdt
end

function jacobian!(jacobian::Matrix{Float64}, t::Float64, x::Vector{Float64}, p::Vector{Float64}) # might be faster if I use SparseMatrixCSC
    N = length(x)
    M = length(p)
    for j in 1:N+M, i in 1:N
        jacobian[i,j]   = p[2]       * ( (mod1(i+1, N) == j) -  (mod1(i-2, N) == j)) * x[mod1(i-1, N)]
                        + p[2]       * (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      *  (mod1(i-1, N) == j)
                        + (N+2 == j) * (x[mod1(i+1, N)]      - x[mod1(i-2, N)])      * x[mod1(i-1, N)]
                        + (N+1 == j)
                        - (i   == j)
    end
    jacobian
end

function hessian!(hessian::Array{Float64, 3}, t::Float64, x::Vector{Float64}, p::Vector{Float64})
    N = length(x)
    M = length(p)
    for k in 1:N+M, j in 1:N+M, i in 1:N
        hessian[i,j,k]  = (N+2 == j) * ( (mod1(i+1, N)==k) -  (mod1(i-2, N)==k)) * x[mod1(i-1, N)]
                        + (N+2 == j) * (x[mod1(i+1, N)]    - x[mod1(i-2, N)])    *  (mod1(i-1, N)==k)
                        + (N+2 == k) * ( (mod1(i+1, N)==j) -  (mod1(i-2, N)==j)) * x[mod1(i-1, N)]
                        + p[2]       * ( (mod1(i+1, N)==j) -  (mod1(i-2, N)==j)) *  (mod1(i-1, N)==k)
                        + (N+2 == k) * (x[mod1(i+1, N)]    - x[mod1(i-2, N)])    *  (mod1(i-1, N)==j)
                        + p[2]       * ( (mod1(i+1, N)==k) -  (mod1(i-2, N)==k)) *  (mod1(i-1, N)==j)
    end
    hessian
end


cost(x, obs, obs_variance) = mapreduce(abs2, +, skipmissing(x .- obs)) / obs_variance / 2. # assuming x[:,1] .= x0; orbit!(dxdt, t, x, p, dt); is already run

innovation_λ(  x, obs, obs_variance) = ismissing(obs) ? 0. : (x - obs)/obs_variance # element-wise innovation of λ

innovation_dλ(dx, obs, obs_variance) = ismissing(obs) ? 0. : dx/obs_variance        # element-wise innovation of dλ

next_x!(dxdt,             dt, t, x, p) = x .+ dxdt!(dxdt, t, x, p) .* dt

next_dx!(jacobian,        dt, t, x, p, dx, dp) = dx .+ jacobian!(jacobian, t, x, p) * Catview(dx, dp) .* dt

@views prev_λ!(jacobian,  dt, t, x, p,         λ,     obs, obs_variance) = λ .+ jacobian!(jacobian, t, x, p)' * λ[1:length(x)] .* dt .+ innovation_λ.(x, obs, obs_variance)

@views prev_dλ!(hessian,  dt, t, x, p, dx, dp, λ, dλ, obs, obs_variance) = prev_λ!(jacobian, t, x, p, dλ) .+ (hessian!(hessian, t, x, p) * CatView(dx, dp))' * λ[1:length(x)] .* dt .+ innovation_dλ.(dx, obs, obs_variance)


@views function orbit!(dxdt, dt, t, x, p)
    for _i in 1:length(t)-1
        x[:,_i+1] .= next_x!(dxdt, dt, t[_i], x[:,_i], p)
    end
    nothing
end

@views function neighboring!(jacobian, dt, t, x, p, dx, dp)
    for _i in 1:length(t)-1
        dx[:,_i+1] .= next_dx!(jacobian, dt, t[_i], x[:,_i], p, dx[:,_i], dp)
    end
    nothing
end

@views function gradient!(jacobian, dt, t, x, p, λ, obs, obs_variance) # assuming x[:,1] .= x0; p .= p; orbit!(dxdt, t, x, p, dt); is already run. λ[:,1] is the gradient.
    λ[:,end] .= innovation_λ(x[:,end], obs[:,end], obs_variance)
    for _i in length(t)-1:-1:1
        λ[:,_i] .= prev_λ!(jacobian, dt, t[_i], x[:,_i], p,               λ[:,_i+1],             obs[:,_i], obs_variance)
    end
    nothing
end

@views function hessian_vector_product!(hessian, dt, t, x, p, dx, dp, λ, dλ, obs, obs_variance) # assuming dx[:,1] .= dx0; dp .= dp; neighboring!(jacobian, dt, t, x, p, dx, dp); is already run. dλ[:,1] is the hessian_vector_product.
    dλ[:,end] .= innovation_dλ(dx[:,end], obs[:,end], obs_variance)
    for _i in length(t)-1:-1:1
        dλ[:,_i] .= prev_dλ(hessian, dt, t[_i], x[:,_i], p, dx[:,_i], dp, λ[:,_i+1], dλ[:,_i+1], obs[:,_i], obs_variance)
    end
    nothing
end


@views function calculate_common!(θ, last_θ, dxdt, dt, t, x, p) # buffering x and p do avoid recalculation of orbit between f! and g!
    if θ != last_θ
        copy!(last_θ, θ)
        N = size(x,1)
        copy!(x[:,1], θ[1:N])
        copy!(p, θ[N+1:end])
        orbit!(dxdt, dt, t, x, p)
    end
end

function f!(θ, dxdt, dt, t, x, p, obs, obs_variance, last_θ)
    calculate_common!(θ, last_θ, dxdt, dt, t, x, p)
    cost(x, obs, obs_variance)
end

@views function g!(θ, ∇θ, dxdt, jacobian, dt, t, x, p, λ, obs, obs_variance, last_θ)
    calculate_common!(θ, last_θ, dxdt, dt, t, x, p)
    gradient!(jacobian, dt, t, x, p, λ, obs, obs_variance)
    copy!(∇θ, λ[:,1])
    nothing
end

function minimize!(initial_θ, dxdt, jacobian, dt, t, x, p, λ, obs, obs_variance)
    last_θ = similar(initial_θ)
    df = TwiceDifferentiable(θ -> f!(θ, dxdt, dt, t, x, p, obs, obs_variance, last_θ),
                             (∇θ, θ) -> g!(θ, ∇θ, dxdt, jacobian, dt, t, x, p, λ, obs, obs_variance, last_θ))
    optimize(df, initial_θ)
end


@views function covariance!(covariance, variance, stddev, hess, jacobian, hessian, dt, t, x, p, dx, dp, λ, dλ, obs, obs_variance)
    _N = size(x,1)
    fill!(dx[:,1], 0.)
    fill!(dp, 0.)
    for i in 1:_N
        dx[i,1] = 1.
        neighboring!(jacobian, dt, t, x, p, dx, dp)
        hessian_vector_product!(hessian, dt, t, x, p, dx, dp, λ, dλ, obs, obs_variance)
        copy!(hess[:,i], dλ[:,1])
        dx[i,1] = 0.
    end
    for i in 1:length(p)
        dp[i] = 1.
        neighboring!(jacobian, dt, t, x, p, dx, dp)
        hessian_vector_product!(hessian, dt, t, x, p, dx, dp, λ, dλ, obs, obs_variance)
        copy!(hess[:,_N+i], dλ[:,1])
        dp[i] = 0.
    end
    covariance .= inv(hess)
    variance .= diag(covariance)
    stddev .= sqrt.(variance)
end
