module Adjoints

using CatViews, NLSolversBase, Optim

export Adjoint, minimize!, covariance!

include("types.jl")
include("../test/Lorenz96/model.jl")
include("adjoint.jl")

end
