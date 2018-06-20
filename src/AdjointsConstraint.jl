module AdjointsConstraint

using CatViews.CatView, NLSolversBase, Optim, Distributions

export Adjoint, Model, AssimilationResults, initialize!, orbit!, gradient!, cost, assimilate!, obs_mean_var!

include("types_constraint.jl")
include("adjoint_constraint.jl")
include("assimilate.jl")

end
