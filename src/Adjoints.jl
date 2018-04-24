module Adjoints

using CatViews, NLSolversBase, Optim, Distributions

export Adjoint, Model, AssimilationResults, initialize!, orbit!, gradient!, cost, assimilate!

include("types.jl")
include("adjoint.jl")
include("assimilate.jl")

end
