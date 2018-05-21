#!/usr/bin/env julia

include("../../util/experiment.jl")
include("model.jl")

const N = 5
const true_p = [8., 1.]
const obs_variance = 1.
const obs_iteration = 5
const dt = 0.01
const spinup = 73.
const T = 1.
const generation_seed = 0
replicates = parse(Int, ARGS[1])
println(replicates)

const pref = "no_loss_data/N_$N/p_$(join(true_p, "_"))/obsvar_$obs_variance/obsiter_$obs_iteration/dt_$dt/spinup_$spinup/T_$T/seed_$generation_seed/"
const observed_file = pref * "observed.tsv"
const true_file = pref * "true/true.tsv"
const observed_pref = pref * "observed/"
observed_files = Tuple([observed_pref * "obsvarseed_$_obsvarseed.tsv" for _obsvarseed in 1:replicates])

model = Model(typeof(dt), N, N+length(true_p), dxdt!, jacobian!, hessian!)
dists = [Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(0., 10.), Uniform(0., 2.)]
twin_experiment!(model, observed_files, obs_variance, dt, true_p, true_file, dists, 20)
