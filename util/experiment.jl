include("../src/Adjoints.jl")
using Adjoints, Distributions, DataFrames, Gadfly

function assimilate_experiment!(model::Model{N,L}, obs, obs_variance, dt, dists, trials=10) where {N,L}
    a = Adjoint(dt, obs_variance, obs, L-N)
    assimilate!(a, model, dists, trials)
end

assimilate_experiment!(model::Model{N,L}, observed_file::String, obs_variance, dt, dists, trials=10) where {N,L} = assimilate_experiment!(model, readdlm(observed_file)', obs_variance, dt, dists, trials)

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

function plot_twin_experiment_result(a::Adjoint{N}, tob) where {N}
    white_panel = Theme(panel_fill="white")
    p_stack = Array{Gadfly.Plot}(0)
    t = collect(0.:a.dt:a.steps*a.dt)
    for _i in 1:N
        df_tob = DataFrame(t=t, x=tob[_i,:], data_type="true orbit")
        _mask = isfinite.(a.obs[_i,:])
        df_obs = DataFrame(t=t[_mask], x=a.obs[_i,:][_mask], data_type="observed")
        df_assim = DataFrame(t=t, x=a.x[_i,:], data_type="assimilated")
        df_all = vcat(df_tob, df_obs, df_assim)
        p_stack = vcat(p_stack, Gadfly.plot(df_all, x="t", y="x", color="data_type", Geom.line, Scale.color_discrete_manual("blue", "red", "green"), Guide.colorkey(title=""), Guide.xlabel("<i>t</i>"), Guide.ylabel("<i>x<sub>$_i</sub></i>"), white_panel))
    end
    draw(PDF("lorenz.pdf", 24cm, 40cm), vstack(p_stack))
    nothing
end

function twin_experiment!(model::Model{N,L,T}, obs::AbstractMatrix{T}, obs_variance::T, dt::T, true_params::AbstractVector{T}, tob::AbstractMatrix{T}, dists, trials=10) where {N,L,T<:AbstractFloat}
    a = Adjoint(dt, obs_variance, obs, L-N)
    assimres, minres = assimilate!(a, model, dists, trials)
    print_twin_experiment_result(assimres, minres.minimum, true_params, tob)
    plot_twin_experiment_result(a, tob)
end

function twin_experiment!(model::Model{N,L,T}, observed_file::String, obs_variance::T, dt::T, true_params::AbstractVector{T}, true_file::String, dists, trials=10) where {N,L,T<:AbstractFloat}
    obs = readdlm(observed_file)'
    tob = readdlm(true_file)'
    twin_experiment!(model, obs, obs_variance, dt, true_params, tob, dists, trials)
end
