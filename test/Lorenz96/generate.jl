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
const x0_dists = [Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.)]
const replicates = 128

model = Model(typeof(dt), N, N+length(true_p), dxdt!, jacobian!, hessian!)
generate_data_no_loss!(model, true_p, obs_variance, obs_iteration, dt, spinup, T, generation_seed, x0_dists, replicates)
