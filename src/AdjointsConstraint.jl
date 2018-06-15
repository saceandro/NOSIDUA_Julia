module AdjointsConstraint

using CatViews.CatView, NLSolversBase, Optim, Distributions

export Adjoint, Model, AssimilationResults, initialize!, orbit!, gradient!, cost, assimilate!

include("types_constraint.jl")
include("adjoint_constraint.jl")
include("assimilate.jl")

end
