module Adjoints

using CatViews, NLSolversBase, Optim

export Adjoint, minimize!, covariance!

include("types.jl")
include("adjoint.jl")

end
