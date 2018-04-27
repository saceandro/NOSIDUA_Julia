include("../src/Adjoints.jl")
using Adjoints, Distributions, DataFrames, Gadfly

# function assimilate_experiment!(model::Model{N,L}, obs, obs_variance, dt, dists, trials=10) where {N,L}
#     a = Adjoint(dt, obs_variance, obs, L-N)
#     assimilate!(a, model, dists, trials)
# end
#
# assimilate_experiment!(model::Model{N,L}, observed_file::String, obs_variance, dt, dists, trials=10) where {N,L} = assimilate_experiment!(model, readdlm(observed_file)', obs_variance, dt, dists, trials)

@views function print_twin_experiment_result(assimilation_results, minimum, true_params, tob)
    println("mincost:\t", minimum)
    println("θ:\t", assimilation_results.θ)
    println("ans:\t", CatViews.CatView(tob[:,1], true_params))
    println("diff:\t", assimilation_results.θ .- CatViews.CatView(tob[:,1], true_params))
    if !(isnull(assimilation_results.stddev))
        println("CI:\t", get(assimilation_results.stddev))
    end
    nothing
end

function plot_twin_experiment_result(a::Adjoint{N}, tob, stddev) where {N}
    white_panel = Theme(panel_fill="white")
    p_stack = Array{Gadfly.Plot}(0)
    t = collect(0.:a.dt:a.steps*a.dt)
    for _i in 1:N
        df_tob = DataFrame(t=t, x=tob[_i,:], data_type="true orbit")
        _mask = isfinite.(a.obs[_i,:,1])
        df_obs = DataFrame(t=t[_mask], x=a.obs[_i,:,1][_mask], data_type="observed") # plot one replicate for example
        xmin = similar(a.x[_i,:], a.steps+1)
        xmin .= NaN
        xmax = similar(a.x[_i,:], a.steps+1)
        xmax .= NaN
        xmin[1] = a.x[_i,1]-stddev[_i]
        xmax[1] = a.x[_i,1]+stddev[_i]
        df_assim = DataFrame(t=t, x=a.x[_i,:], xmin=xmin, xmax=xmax, data_type="assimilated")
        p_stack = vcat(p_stack,
        Gadfly.plot(
        layer(df_tob, x="t", y="x", color=:data_type, Geom.line),
        layer(df_obs, x="t", y="x", color=:data_type, Geom.point),
        layer(df_assim, x="t", y="x", ymin="xmin", ymax="xmax", color=:data_type, Geom.line, Geom.errorbar),
        Guide.xlabel("<i>t</i>"),
        Guide.ylabel("<i>x<sub>$_i</sub></i>"),
        white_panel))
    end
    draw(PDF("lorenz.pdf", 24cm, 40cm), vstack(p_stack))
    nothing
end

function twin_experiment!(model::Model{N,L,T}, obs::AbstractArray{T,3}, obs_variance::T, dt::T, true_params::AbstractVector{T}, tob::AbstractMatrix{T}, dists, trials=10) where {N,L,T<:AbstractFloat}
    a = Adjoint(dt, obs_variance, obs, L-N)
    assimres, minres = assimilate!(a, model, dists, trials)
    print_twin_experiment_result(assimres, minres.minimum, true_params, tob)
    plot_twin_experiment_result(a, tob, get(assimres.stddev))
end

function twin_experiment!(model::Model{N,L,T}, observed_files::Tuple, obs_variance::T, dt::T, true_params::AbstractVector{T}, true_file::String, dists, trials=10) where {N,L,T<:AbstractFloat}
    obs1 = readdlm(observed_files[1])'
    steps = size(obs1,2)
    K = length(observed_files)
    obs = Array{T}(N, steps, K)
    obs[:,:,1] .= obs1
    for _replicate in 2:K
        obs[:,:,_replicate] .= readdlm(observed_files[_replicate])'
    end
    twin_experiment!(model, obs, obs_variance, dt, true_params, readdlm(true_file)', dists, trials)
end

@views function generate_data!(model::Model{N}, true_params, obs_variance, obs_iteration, dt, spinup, T, generation_seed, x0_dists, replicates) where {N}
    pref = "data/N_$N/p_$(join(true_p, "_"))/obsvar_$obs_variance/obsiter_$obs_iteration/dt_$dt/spinup_$spinup/T_$T/seed_$generation_seed/"
    true_pref = pref * "true/"
    mkpath(true_pref)
    srand(generation_seed)
    x0 = rand.(x0_dists)
    a = Adjoint(dt, T, obs_variance, x0, true_params)
    orbit!(a, model)
    writedlm(true_pref * "true.tsv", a.x')
    d = Normal(0., sqrt(obs_variance))
    steps_iter = a.steps/obs_iteration
    for _obsvarseed in 1:replicates
        obs_pref = pref * "observed/"
        mkpath(obs_pref)
        srand(_obsvarseed)
        obs = a.x .+ rand(d, N, a.steps+1)
        maskall = rand(1:N, N, a.steps+1)
        for _i in 1:obs_iteration:a.steps
            for _j in 1:N
                if maskall[_j, _i] == 1
                    obs[_j, _i] = NaN
                end
            end
            for _k in 1:obs_iteration-1
                obs[:, _i + _k] .= NaN
            end
        end
        writedlm(obs_pref * "obsvarseed_$_obsvarseed.tsv", obs')
    end
end
