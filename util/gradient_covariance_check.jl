include("../src/Adjoints.jl")
using Adjoints, Distributions

orbit_cost!(a, m) = (orbit!(a, m); cost(a))

orbit_gradient!(a, m) = (orbit!(a, m); gradient!(a, m); a.λ[:,1])

function numerical_gradient!(a::Adjoint{N,L,K,T}, m::Model{N,L}, h) where {N,L,K,T}
    gr = Vector{T}(L)
    c = orbit_cost!(a, m)
    for _i in 1:N
        a.x[_i,1] += h
        gr[_i] = (orbit_cost!(a, m) - c)/h
        a.x[_i,1] -= h
    end
    for _i in N+1:L
        a.p[_i-N] += h
        gr[_i] = (orbit_cost!(a, m) - c)/h
        a.p[_i-N] -= h
    end
    orbit!(a, m)
    gr
end

function numerical_covariance!(a::Adjoint{N,L,K,T}, m::Model{N,L}, h) where {N,L,K,T}
    hessian = Matrix{T}(L, L)
    gr = orbit_gradient!(a, m)
    for _i in 1:N
        a.x[_i,1] += h
        hessian[:,_i] .= (orbit_gradient!(a, m) .- gr)/h
        a.x[_i,1] -= h
    end
    for _i in N+1:L
        a.p[_i-N] += h
        hessian[:,_i] .= (orbit_gradient!(a, m) .- gr)/h
        a.p[_i-N] -= h
    end
    orbit_gradient!(a, m)

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

@views function gradient_covariance_check!(model::Model{N,L,T}, observed_files, obs_variance, dt, dists, trials=20, h=0.001) where {N,L,T}
    obs1 = readdlm(observed_files[1])'
    steps = size(obs1,2)
    K = length(observed_files)
    obs = Array{T}(N, steps, K)
    copy!(obs[:,:,1], obs1)
    for _replicate in 2:K
        obs[:,:,_replicate] .= readdlm(observed_files[_replicate])'
    end
    a = Adjoint(dt, obs_variance, obs, L-N)
    x0_p = rand.(dists)
    initialize!(a, x0_p)
    gr_ana = orbit_gradient!(a, model)
    gr_num = numerical_gradient!(a, model, h)
    println("analytical gradient:\t", gr_ana)
    println("numerical gradient:\t", gr_num)
    diff = gr_ana .- gr_num
    println("absolute difference:\t", diff)
    println("max absolute difference:\t", maximum(abs, diff))
    if !any(gr_num == 0)
        rel_diff = diff ./ gr_num
        println("relative difference:\t", rel_diff)
        println("max relative difference:\t", maximum(abs, rel_diff))
    end

    res_ana, minres = assimilate!(a, model, dists, trials)
    res_num = numerical_covariance!(a, model, h)

    # if !isnull(res_ana.covariance) && !isnull(res_num.covariance)
    #     cov_ana = get(res_ana.covariance)
    #     cov_num = get(res_num.covariance)
    #     println("analytical covariance:")
    #     println(cov_ana)
    #     println("numerical covariance:")
    #     println(cov_num)
    #     diff = cov_ana .- cov_num
    #     println("absolute difference:")
    #     println(diff)
    #     println("max absolute difference: ", maximum(abs, diff))
    #     if !any(res_num.covariance == 0)
    #         rel_diff = diff ./ cov_num
    #         println("relative difference")
    #         println(rel_diff)
    #         println("max_relative_difference:\t", maximum(abs, rel_diff))
    #     end
    # end

    if !isnull(res_ana.stddev) && !isnull(res_num.stddev)
        cov_ana = get(res_ana.stddev)
        cov_num = get(res_num.stddev)
        println("analytical stddev:\t", cov_ana)
        println("numerical stddev:\t", cov_num)
        diff = cov_ana .- cov_num
        println("absolute difference:\t", diff)
        println("max absolute difference:\t", maximum(abs, diff))
        if !any(res_num.stddev == 0)
            rel_diff = diff ./ cov_num
            println("relative difference:\t", rel_diff)
            println("max_relative_difference:\t", maximum(abs, rel_diff))
        end
    end
end
