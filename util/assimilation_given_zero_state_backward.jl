nanzero(x) = isnan(x) ? zero(x) : x

digits10(x) = map(x -> @sprintf("%.10f", x), x)
join_digits10(x) = join(digits10(x), "_")

@views function obs_mean_var!(a::Adjoint{N}, m::Model{N}, obs) where {N}
    all!(a.finite, isfinite.(obs))
    a.obs_mean = reshape(mean(obs, 3), N, a.steps+1)
    # a.obs_filterd_var = reshape(sum(reshape(reshape(var(obs, 3; corrected=false), N, a.steps+1)[a.finite], N, :), 2), N)
    for _j in 1:N
        a.obs_filterd_var[_j] = mapreduce(nanzero, +, var(obs[_j,:,:], 2; corrected=false))
        a.Nobs[_j] .+= count(isfinite.(obs[_j,:,:]))
    end
    nothing
end

@views function write_twin_experiment_result(dir, assimilation_results::AssimilationResults{M}, minimum, true_params, tob) where {M}
    mkpath(dir)
    if isnull(assimilation_results.p)
        writedlm(dir * "estimates.tsv", reshape(CatView(fill(NaN, M), fill(NaN, M)), M, 2))
        return nothing
    end

    println(STDERR, "mincost:\t", minimum)
    println(STDERR, "p:\t", get(assimilation_results.p))
    println(STDERR, "ans:\t", true_params)
    diff = get(assimilation_results.p) .- true_params
    println(STDERR, "diff:\t", diff)
    println(sqrt(mapreduce(abs2, +, diff) / M)) # output RSME to STDOUT
    if isnull(assimilation_results.stddev)
        writedlm(dir * "estimates.tsv", reshape(CatView(diff, fill(NaN, M)), M, 2))
    else
        println(STDERR, "CI:\t", get(assimilation_results.stddev))
        writedlm(dir * "estimates.tsv", reshape(CatView(diff, get(assimilation_results.stddev)), M, 2))
    end
    println(STDERR, "obs variance:\t", get(assimilation_results.obs_variance))
    nothing
end

@views function write_assimilation_result(dir, file, assimilation_results::AssimilationResults{M}, minimum, parameters) where {M}
    mkpath(dir)

    f = open(dir * "$(file)_table.tsv", "w")
    println(f, "\t", join(parameters, "\t"))

    if isnull(assimilation_results.p)
        writedlm(dir * "$file.tsv", reshape(CatView(fill(NaN, M), fill(NaN, M)), M, 2))
        return nothing
    end

    println(STDERR, "mincost:\t", minimum)
    println(STDERR, "p:\t", get(assimilation_results.p))
    println(f, "parameter\t", join(exp.(get(assimilation_results.p)), "\t"))
    println(f, "log(parameter)\t", join(get(assimilation_results.p), "\t"))

    if !isnull(assimilation_results.precision)
        println(STDERR, "precision:\t", get(assimilation_results.precision))
    end
    if isnull(assimilation_results.stddev)
        writedlm(dir * "$file.tsv", reshape(CatView(get(assimilation_results.p), fill(NaN, M)), M, 2))
    else
        println(STDERR, "CI:\t", get(assimilation_results.stddev))
        println(f, "CI(log(parameter))\t", join(get(assimilation_results.stddev), "\t"))
        writedlm(dir * "$file.tsv", reshape(CatView(get(assimilation_results.p), get(assimilation_results.stddev)), M, 2))
    end
    close(f)
    println(STDERR, "obs variance:\t", get(assimilation_results.obs_variance))
    writedlm(dir * "$(file)_obsvar.tsv", get(assimilation_results.obs_variance))

    if !isnull(assimilation_results.precision)
        writedlm(dir * "$(file)_precision.tsv", get(assimilation_results.precision))

        maxabs = maximum(abs.(get(assimilation_results.precision)))
        pl = spy(get(assimilation_results.precision), Scale.x_discrete(; labels= i->parameters[i]), Scale.y_discrete(; labels= i->parameters[i]), Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"), minvalue=-maxabs, maxvalue=maxabs), Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey("precision"), Theme(panel_fill="white"))
        draw(PDF(dir * "$(file)_precision_heatmap.pdf", 24cm, 24cm), pl)
    end
    if !isnull(assimilation_results.covariance)
        writedlm(dir * "$(file)_covariance.tsv", get(assimilation_results.covariance))

        maxabs = maximum(abs.(get(assimilation_results.covariance)))
        pl = spy(get(assimilation_results.covariance), Scale.x_discrete(; labels= i->parameters[i]), Scale.y_discrete(; labels= i->parameters[i]), Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"), minvalue=-maxabs, maxvalue=maxabs), Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey("covariance"), Theme(panel_fill="white"))
        draw(PDF(dir * "$(file)_covariance_heatmap.pdf", 24cm, 24cm), pl)

        if !isnull(assimilation_results.stddev)
            inv_stddev_diagm = diagm(inv.(get(assimilation_results.stddev)))
            corr = inv_stddev_diagm * get(assimilation_results.covariance) * inv_stddev_diagm
            writedlm(dir * "$(file)_cor.tsv", corr)
            pl = spy(corr, Scale.x_discrete(; labels= i->parameters[i]), Scale.y_discrete(; labels= i->parameters[i]), Scale.color_continuous(colormap=Scale.lab_gradient("blue", "white", "red"), minvalue=-1., maxvalue=1.), Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey("correlation"), Theme(panel_fill="white"))
            draw(PDF(dir * "$(file)_cor_heatmap.pdf", 24cm, 24cm), pl)
        end
    end

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

@views function twin_experiment!(
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
    model = Model(typeof(dt), N, L, dxdt!, jacobian!, hessian!)
    srand(generation_seed)
    dists = [Uniform(initial_lower_bounds[i], initial_upper_bounds[i]) for i in 1:L]
    x0 = rand.(view(dists, 1:N))
    a = Adjoint(dt, duration, copy(obs_variance), x0, copy(true_params), replicates)
    orbit!(a, model)
    srand(hash([true_params, initial_lower_bounds, initial_upper_bounds, obs_variance, obs_iteration, dt, spinup, duration, generation_seed, trials, replicates, iter]))
    d = Normal.(0., sqrt.(obs_variance))
    for _replicate in 1:replicates
        for _i in 1:N
            a.obs[_i,:,_replicate] .= a.x[_i,:] .+ rand(d[_i], a.steps+1)
        end
        for _i in 1:obs_iteration:a.steps
            for _k in 1:obs_iteration-1
                a.obs[:, _i + _k, _replicate] .= NaN
            end
        end
    end
    dir *= "/true_params_$(join(true_params, "_"))/initial_lower_bounds_$(join(initial_lower_bounds, "_"))/initial_upper_bounds_$(join(initial_upper_bounds, "_"))/spinup_$spinup/generation_seed_$generation_seed/trials_$trials/obs_variance_$(join(obs_variance, "_"))/obs_iteration_$obs_iteration/dt_$dt/duration_$duration/replicates_$replicates/iter_$iter/"
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
    twin_experiment!(dir, a, model, true_params, dists, trials)
end


function twin_experiment!(outdir::String, a::Adjoint{N,L,K,T}, model::Model{N,L,T}, true_params::AbstractVector{T}, initial_lower_bounds::AbstractVector{T}, initial_upper_bounds::AbstractVector{T}, dists, trials=10) where {N,L,K,T<:AbstractFloat}
    tob = deepcopy(a.x) # fixed bug. copy() of >=2 dimensional array is implemented as reference. Thus, deepcopy() is needed.
    println(STDERR, "====================================================================================================================")
    println(STDERR, outdir)
    assimres, minres = assimilate!(a, model, initial_lower_bounds, initial_upper_bounds, dists, trials)
    if (isnull(assimres.p))
        write_twin_experiment_result(outdir, assimres, zero(T), true_params, tob)
    else
        write_twin_experiment_result(outdir, assimres, minres.minimum, true_params, tob)
    end
end

function assimilation_procedure!(outdir::String, file::String, a::Adjoint{N,L,K,T}, model::Model{N,L,T}, initial_lower_bounds::AbstractVector{T}, initial_upper_bounds::AbstractVector{T}, dists, parameters, trials=10) where {N,L,K,T<:AbstractFloat}
    println(STDERR, "====================================================================================================================")
    println(STDERR, outdir)
    assimres, minres = assimilate!(a, model, initial_lower_bounds, initial_upper_bounds, dists, trials)
    if (isnull(assimres.p))
        write_assimilation_result(outdir, file, assimres, zero(T), parameters)
    else
        write_assimilation_result(outdir, file, assimres, minres.minimum, parameters)
    end
end

@views function generate_true_data!(pref::String, model::Model{N}, true_params, dt, spinup, T, generation_seed, x0_dists) where {N}
    srand(generation_seed)
    x0 = rand.(x0_dists)
    a = Adjoint(dt, T, 1., x0, true_params) # 1. means obs_variance. defined as 1 though not used.
    orbit!(a, model)
    mkpath(pref)
    writedlm(pref * "seed_$generation_seed.tsv", a.x')
end

function plot_twin_experiment_result_wo_errorbar_observation(dir, file, a::Adjoint{N,L,K}, m::Model, obs) where {N,L,K}
    white_panel = Theme(panel_fill="white")
    p_stack = Array{Gadfly.Plot}(0)
    t = collect(0.:a.dt:a.steps*a.dt)
    for _i in 1:N
        _mask = isfinite.(view(reshape(obs, N, :), _i, :))
        df_obs = DataFrame(t=view(repeat(t; outer=[K]), :)[_mask], x=view(reshape(obs, N, :), _i, :)[_mask], data_type="observed")
        df_assim = DataFrame(t=t, x=m.observation.(a.x[_i,:]), data_type="assimilated")
        p_stack = vcat(p_stack,
        Gadfly.plot(
        layer(df_assim, x="t", y="x", color=:data_type, Geom.line),
        layer(df_obs, x="t", y="x", color=:data_type, Geom.point),
        Guide.xlabel("<i>t</i>"),
        Guide.ylabel("<i>x<sub>$_i</sub></i>"),
        white_panel))
    end
    draw(PDF(dir * "$(file)_observation.pdf", 24cm, 40cm), vstack(p_stack))
    nothing
    # set_default_plot_size(24cm, 40cm)
    # vstack(p_stack)
end
