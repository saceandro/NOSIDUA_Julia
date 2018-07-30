module AdjointsGivenZeroStateBackward

using CatViews.CatView, Distributions, LineSearches, Optim

export Adjoint, Model, AssimilationResults, initialize_p!, orbit!, gradient!, cost, assimilate!, obs_mean_var!, negative_log_likelihood

include("types_given_zero_state_backward.jl")
include("adjoint_given_zero_state_backward.jl")
include("assimilate_given_zero_state.jl")

end
