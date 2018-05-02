include("../../src/Adjoints.jl")

module Replicate

using Adjoints, Distributions

export julia_main

include("model.jl")
include("../../util/experiment_ccompile.jl")

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    N = 5
    true_p = [8., 1.]
    obs_variance = 1.
    obs_iteration = 5
    dt = 0.01
    spinup = 73.
    T = 1.
    generation_seed = 0

    pref = "no_loss_data/N_$N/p_$(join(true_p, "_"))/obsvar_$obs_variance/obsiter_$obs_iteration/dt_$dt/spinup_$spinup/T_$T/seed_$generation_seed/"
    true_file = pref * "true/true.tsv"
    observed_pref = pref * "observed/"

    model = Model(typeof(dt), N, N+length(true_p), dxdt!, jacobian!, hessian!)
    dists = [Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(0., 10.), Uniform(0., 2.)]
    replicates = parse(Int, ARGS[1])
    observed_files = Tuple([observed_pref * "obsvarseed_$_obsvarseed.tsv" for _obsvarseed in 1:replicates])
    twin_experiment!(model, observed_files, obs_variance, dt, true_p, true_file, dists, 20)
    return 0
end

end
