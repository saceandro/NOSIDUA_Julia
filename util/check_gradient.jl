using DataFrames, Gadfly

@views function write_twin_experiment_result(dir, assimilation_results, minimum, true_params, tob)
    mkpath(dir)
    L = length(assimilation_results.θ)
    println(STDERR, "mincost:\t", minimum)
    println(STDERR, "θ:\t", assimilation_results.θ)
    println(STDERR, "ans:\t", CatView(tob[:,1], true_params))
    diff = assimilation_results.θ .- CatView(tob[:,1], true_params)
    println(STDERR, "diff:\t", diff)
    println(sqrt(mapreduce(abs2, +, diff) / L)) # output RSME to STDOUT
    if isnull(assimilation_results.stddev)
        writedlm(dir * "estimates.tsv", reshape(CatView(diff, fill(NaN, L)), L, 2))
    else
        println(STDERR, "CI:\t", get(assimilation_results.stddev))
        writedlm(dir * "estimates.tsv", reshape(CatView(diff, get(assimilation_results.stddev)), L, 2))
    end
    println(STDERR, "obs variance:\t", assimilation_results.obs_variance)
    nothing
end

# function twin_experiment!(model::Model{N,L,T}, observed_files::Tuple, obs_variance::T, dt::T, true_params::AbstractVector{T}, true_file::String, dists, trials=10) where {N,L,T<:AbstractFloat}
#     obs1 = readdlm(observed_files[1])'
#     steps = size(obs1,2)
#     K = length(observed_files)
#     obs = Array{T}(N, steps, K)
#     obs[:,:,1] .= obs1
#     for _replicate in 2:K
#         obs[:,:,_replicate] .= readdlm(observed_files[_replicate])'
#     end
#     twin_experiment!(model, obs, obs_variance, dt, true_params, readdlm(true_file)', dists, trials)
# end

# function twin_experiment!(outdir::String, model::Model{N,L,T}, obs::AbstractArray{T,3}, obs_variance::T, dt::T, true_params::AbstractVector{T}, tob::AbstractMatrix{T}, dists, trials=10) where {N,L,T<:AbstractFloat}
#     a = Adjoint(dt, obs_variance, obs, L-N)
#     assimres, minres = assimilate!(a, model, dists, trials)
#     write_twin_experiment_result(outdir, assimres, minres.minimum, true_params, tob)
# end

# twin_experiment!(outdir::String, model::Model{N,L,T}, obs_variance::T, obs_iteration::Int, dt::T, true_params::AbstractVector{T}, true_file::String, dists, replicates::Int, iter::Int, trials=10) where {N,L,T} = twin_experiment!(outdir, model, obs_variance, obs_iteration, dt, true_params, readdlm(true_file)', dists, replicates, iter, trials)

function twin_experiment!(
    dxdt!::Function,
    jacobian!::Function,
    hessian!::Function;
    dir = nothing,
    true_params = nothing,
    initial_lower_bounds = nothing,
    initial_upper_bounds = nothing,
    obs_variance = nothing,
    obs_iteration = nothing,
    dt = nothing,
    spinup = nothing,
    duration = nothing,
    generation_seed = nothing,
    trials = nothing,
    replicates = nothing,
    iter = nothing
    )

    L = length(initial_lower_bounds)
    N = L - length(true_params)
    model = Model(typeof(obs_variance), N, L, dxdt!, jacobian!, hessian!)
    srand(generation_seed)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L]
    x0 = rand.(view(dists, 1:N))
    a = Adjoint(dt, duration, obs_variance, x0, copy(true_params), replicates)
    orbit!(a, model)
    srand(hash([true_params, initial_lower_bounds, initial_upper_bounds, obs_variance, obs_iteration, dt, spinup, duration, generation_seed, trials, replicates, iter]))
    d = Normal(0., sqrt(obs_variance))
    for _replicate in 1:replicates
        a.obs[:,:,_replicate] .= a.x .+ rand(d, N, a.steps+1)
        for _i in 1:obs_iteration:a.steps
            for _k in 1:obs_iteration-1
                a.obs[:, _i + _k, _replicate] .= NaN
            end
        end
    end
    dir *= "/true_params_$(join(true_params, "_"))/initial_lower_bounds_$(join(initial_lower_bounds, "_"))/initial_upper_bounds_$(join(initial_upper_bounds, "_"))/spinup_$spinup/generation_seed_$generation_seed/trials_$trials/obs_variance_$obs_variance/obs_iteration_$obs_iteration/dt_$dt/duration_$duration/replicates_$replicates/iter_$iter/"
    twin_experiment!(dir, a, model, true_params, initial_lower_bounds, initial_upper_bounds, dists, trials)
end

function twin_experiment_logging!( # twin experiment with true and obs data logging
    dxdt!::Function,
    jacobian!::Function,
    hessian!::Function;
    dir = nothing,
    true_params = nothing,
    initial_lower_bounds = nothing,
    initial_upper_bounds = nothing,
    obs_variance = nothing,
    obs_iteration = nothing,
    dt = nothing,
    spinup = nothing,
    duration = nothing,
    generation_seed = nothing,
    trials = nothing,
    replicates = nothing,
    iter = nothing
    )

    L = length(initial_lower_bounds)
    N = L - length(true_params)
    model = Model(typeof(obs_variance), N, L, dxdt!, jacobian!, hessian!)
    srand(generation_seed)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L]
    x0 = rand.(view(dists, 1:N))
    a = Adjoint(dt, duration, obs_variance, x0, copy(true_params), replicates)
    orbit!(a, model)
    dir *= "/true_params_$(join(true_params, "_"))/initial_lower_bounds_$(join(initial_lower_bounds, "_"))/initial_upper_bounds_$(join(initial_upper_bounds, "_"))/spinup_$spinup/generation_seed_$generation_seed/trials_$trials/obs_variance_$obs_variance/obs_iteration_$obs_iteration/dt_$dt/duration_$duration/replicates_$replicates/iter_$iter/"
    mkpath(dir)
    writedlm(dir * "true.tsv", a.x')
    srand(hash([true_params, initial_lower_bounds, initial_upper_bounds, obs_variance, obs_iteration, dt, spinup, duration, generation_seed, trials, replicates, iter]))
    d = Normal(0., sqrt(obs_variance))
    for _replicate in 1:replicates
        a.obs[:,:,_replicate] .= a.x .+ rand(d, N, a.steps+1)
        for _i in 1:obs_iteration:a.steps
            for _k in 1:obs_iteration-1
                a.obs[:, _i + _k, _replicate] .= NaN
            end
        end
        writedlm(dir * "observed$_replicate.tsv", view(a.obs, :, :, _replicate)')
    end
    twin_experiment!(dir, a, model, true_params, initial_lower_bounds, initial_upper_bounds, dists, trials)
end


function twin_experiment!(outdir::String, a::Adjoint{N,L,K,T}, model::Model{N,L,T}, true_params::AbstractVector{T}, initial_lower_bounds::AbstractVector{T}, initial_upper_bounds::AbstractVector{T}, dists, trials=10) where {N,L,K,T<:AbstractFloat}
    tob = deepcopy(a.x) # fixed bug. copy() of >=2 dimensional array is implemented as reference. Thus, deepcopy() is needed.
    println(STDERR, "====================================================================================================================")
    println(STDERR, outdir)
    assimres, minres = assimilate!(a, model, initial_lower_bounds, initial_upper_bounds, dists, trials)
    write_twin_experiment_result(outdir, assimres, minres.minimum, true_params, tob)
end

@views function generate_true_data!(pref::String, model::Model{N}, true_params, dt, spinup, T, generation_seed, x0_dists) where {N}
    srand(generation_seed)
    x0 = rand.(x0_dists)
    a = Adjoint(dt, T, 1., x0, true_params) # 1. means obs_variance. defined as 1 though not used.
    orbit!(a, model)
    mkpath(pref)
    writedlm(pref * "seed_$generation_seed.tsv", a.x')
end

orbit_cost!(a, m) = (orbit!(a, m); cost(a))

orbit_gradient!(a, m, gr) = (orbit!(a, m); gradient!(a, m, gr))

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

@views function numerical_covariance!(a::Adjoint{N,L,K,T}, m::Model{N,L}, h) where {N,L,K,T}
    hessian = Matrix{T}(L, L)
    gr = Vector{T}(L)
    orbit_gradient!(a, m, gr)
    for _i in 1:N
        a.x[_i,1] += h
        orbit_gradient!(a, m, hessian[:,_i])
        hessian[:,_i] .-= gr
        hessian[:,_i] ./= h
        a.x[_i,1] -= h
    end
    for _i in N+1:L
        a.p[_i-N] += h
        orbit_gradient!(a, m, hessian[:,_i])
        hessian[:,_i] .-= gr
        hessian[:,_i] ./= h
        a.p[_i-N] -= h
    end
    orbit_gradient!(a, m, gr)

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

@views function gradient_covariance_check!(
    dxdt!::Function,
    jacobian!::Function,
    hessian!::Function;
    dir = nothing,
    true_params = nothing,
    initial_lower_bounds = nothing,
    initial_upper_bounds = nothing,
    obs_variance = nothing,
    obs_iteration = nothing,
    dt = nothing,
    spinup = nothing,
    duration = nothing,
    generation_seed = nothing,
    trials = nothing,
    replicates = nothing,
    iter = nothing,
    h = 0.001)

    L = length(initial_lower_bounds)
    N = L - length(true_params)
    model = Model(typeof(obs_variance), N, L, dxdt!, jacobian!, hessian!)
    srand(generation_seed)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L]
    x0 = rand.(view(dists, 1:N))
    a = Adjoint(dt, duration, obs_variance, x0, copy(true_params), replicates)
    orbit!(a, model)
    tob = deepcopy(a.x)
    srand(hash([true_params, initial_lower_bounds, initial_upper_bounds, obs_variance, obs_iteration, dt, spinup, duration, generation_seed, trials, replicates, iter]))
    d = Normal(0., sqrt(obs_variance))
    for _replicate in 1:replicates
        a.obs[:,:,_replicate] .= a.x .+ rand(d, N, a.steps+1)
        for _i in 1:obs_iteration:a.steps
            for _k in 1:obs_iteration-1
                a.obs[:, _i + _k, _replicate] .= NaN
            end
        end
    end

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
    dir *= "/true_params_$(join(true_params, "_"))/initial_lower_bounds_$(join(initial_lower_bounds, "_"))/initial_upper_bounds_$(join(initial_upper_bounds, "_"))/spinup_$spinup/generation_seed_$generation_seed/trials_$trials/obs_variance_$obs_variance/obs_iteration_$obs_iteration/dt_$dt/duration_$duration/replicates_$replicates/iter_$iter/"
    write_twin_experiment_result(dir, res_ana, minres.minimum, true_params, tob)

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
        plot_twin_experiment_result(dir, a, tob, cov_ana)
    else
        plot_twin_experiment_result_wo_errorbar(dir, a, tob)
    end
end

function plot_twin_experiment_result(dir, a::Adjoint{N,L,K}, tob, obs, stddev) where {N,L,K}
    white_panel = Theme(panel_fill="white")
    p_stack = Array{Gadfly.Plot}(0)
    t = collect(0.:a.dt:a.steps*a.dt)
    for _i in 1:N
        df_tob = DataFrame(t=t, x=tob[_i,:], data_type="true orbit")
        # _mask = isfinite.(a.obs[_i,:,1])
        _mask = isfinite.(view(reshape(obs, N, :), _i, :))
        # df_obs = DataFrame(t=t[_mask], x=a.obs[_i,:,1][_mask], data_type="observed") # plot one replicate for example
        df_obs = DataFrame(t=view(repeat(t; outer=[K]), :)[_mask], x=view(reshape(obs, N, :), _i, :)[_mask], data_type="observed")
        xmin = similar(view(a.x, _i, :), a.steps+1)
        xmin .= NaN
        xmax = similar(view(a.x, _i, :), a.steps+1)
        xmax .= NaN
        xmin[1] = a.x[_i,1]-stddev[_i]
        xmax[1] = a.x[_i,1]+stddev[_i]
        df_assim = DataFrame(t=t, x=a.x[_i,:], xmin=xmin, xmax=xmax, data_type="assimilated")
        p_stack = vcat(p_stack,
        Gadfly.plot(
        layer(df_tob, x="t", y="x", color=:data_type, Geom.line),
        layer(df_assim, x="t", y="x", ymin="xmin", ymax="xmax", color=:data_type, Geom.line, Geom.errorbar),
        layer(df_obs, x="t", y="x", color=:data_type, Geom.point),
        Guide.xlabel("<i>t</i>"),
        Guide.ylabel("<i>x<sub>$_i</sub></i>"),
        white_panel))
    end
    draw(PDF(dir * "assimilation.pdf", 24cm, 40cm), vstack(p_stack))
    nothing
    # set_default_plot_size(24cm, 40cm)
    # vstack(p_stack)
end

function plot_twin_experiment_result_wo_errorbar(dir, a::Adjoint{N}, tob) where {N}
    white_panel = Theme(panel_fill="white")
    p_stack = Array{Gadfly.Plot}(0)
    t = collect(0.:a.dt:a.steps*a.dt)
    for _i in 1:N
        df_tob = DataFrame(t=t, x=tob[_i,:], data_type="true orbit")
        # _mask = isfinite.(a.obs[_i,:,1])
        _mask = isfinite.(a.obs[_i,:,:])
        # df_obs = DataFrame(t=t[_mask], x=a.obs[_i,:,1][_mask], data_type="observed") # plot one replicate for example
        df_obs = DataFrame(t=t[_mask], x=a.obs[_i,:,:][_mask], data_type="observed")
        df_assim = DataFrame(t=t, x=a.x[_i,:], data_type="assimilated")
        p_stack = vcat(p_stack,
        Gadfly.plot(
        layer(df_tob, x="t", y="x", color=:data_type, Geom.line),
        layer(df_assim, x="t", y="x", color=:data_type, Geom.line),
        layer(df_obs, x="t", y="x", color=:data_type, Geom.point),
        Guide.xlabel("<i>t</i>"),
        Guide.ylabel("<i>x<sub>$_i</sub></i>"),
        white_panel))
    end
    draw(PDF(dir * "assimilation.pdf", 24cm, 40cm), vstack(p_stack))
    nothing
    # set_default_plot_size(24cm, 40cm)
    # vstack(p_stack)
end
