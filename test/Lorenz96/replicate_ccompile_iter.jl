include("../../src/Adjoints.jl")

module ReplicateIter

using Adjoints, Distributions

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
    seed = 0

    model = Model(typeof(dt), N, N+length(true_params), dxdt!, jacobian!, hessian!)
    dists = [Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(-10., 10.), Uniform(0., 10.), Uniform(0., 2.)]
    replicates = parse(Int, ARGS[1])
    twin_experiment_iter!(outdir, model, true_params, obs_variance, obs_iteration, dt, spinup, T, seed, replicates, dists, 50, 10)
    return 0
end

end
