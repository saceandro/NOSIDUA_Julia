include("../../util/gradient_covariance_check.jl")
include("model.jl")

const N = 5
const true_p = [8., 1.]
const obs_variance = 1.
const obs_iteration = 5
const dt = 0.01
const spinup = 73.
const T = 1.
const generation_seed = 0

const pref = "../../adjoint/data/Lorenz96/N_$N/p_$(join(true_p, "_"))/obsvar_$obs_variance/obsiter_$obs_iteration/dt_$dt/spinup_$spinup/T_$T/seed_$generation_seed/"
const observed_file = pref * "observed.tsv"
const true_file = pref * "true.tsv"

model = Model(typeof(dt), N, N+length(true_p), dxdt!, jacobian!, hessian!)
dists = [Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(0., 10.), Uniform(0., 2.)]
gradient_covariance_check!(model, observed_file, obs_variance, dt, dists, 20, 0.0001)
