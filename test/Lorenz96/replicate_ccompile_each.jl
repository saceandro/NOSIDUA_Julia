include("../../src/Adjoints.jl")

module ReplicateIter

using CatViews.CatView, Adjoints, Distributions

export julia_main

include("model.jl")
include("../../util/experiment_ccompile.jl")

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    outdir = "result/"
    N = 5
    true_params = [8., 1.]
    obs_variance = 1.
    obs_iteration = 5
    dt = 0.01
    spinup = 73.
    T = 1.
    generation_seed = 0
    trials = 50

    pref = "true_data/N_$N/p_$(join(true_params, "_"))/dt_$dt/spinup_$spinup/T_$T/"

    model = Model(typeof(dt), N, N+length(true_params), dxdt!, jacobian!, hessian!)
    dists = [Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(0., 10.), Uniform(0., 2.)]

    generate_true_data!(pref, model, true_params, dt, spinup, T, generation_seed, view(dists, 1:N))

    replicates = parse(Int, ARGS[1])
    iter = parse(Int, ARGS[2])

    true_file = pref * "seed_$(generation_seed)_forreplicate_$(replicates)_foriter_$(iter).tsv"

    twin_experiment!(outdir, model, obs_variance, obs_iteration, dt, true_params, true_file, dists, replicates, iter, trials)
    return 0
end

end
