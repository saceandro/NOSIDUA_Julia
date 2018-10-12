module AdjointsBackward

using CatViews.CatView, Distributions, LineSearches, Optim

export Adjoint, Model, AssimilationResults, initialize!, orbit!, gradient!, cost, assimilate!, obs_mean_var!, negative_log_likelihood

include("types_backward.jl")
include("adjoint_backward.jl")
include("assimilate_backward.jl")

end
