module AdjointsGivenZeroState

using CatViews.CatView, NLSolversBase, Optim, Distributions

export Adjoint, Model, AssimilationResults, initialize_p!, orbit!, gradient!, cost, assimilate!, obs_mean_var!

include("types_given_zero_state.jl")
include("adjoint_given_zero_state.jl")
include("assimilate_given_zero_state.jl")

end
