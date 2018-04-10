module Adjoints

using CatViews, NLSolversBase, Optim

export Adjoint, Model, minimize!, covariance!, orbit_gradient!, numerical_gradient!, numerical_hessian!

include("types.jl")
include("adjoint.jl")

end
