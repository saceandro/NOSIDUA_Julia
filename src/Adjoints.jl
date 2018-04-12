module Adjoints

using CatViews, NLSolversBase, Optim
import Main: dxdt!, jacobian!, hessian!

export Adjoint, minimize!, covariance!, orbit_gradient!, numerical_gradient!, numerical_hessian!

include("types.jl")
include("adjoint.jl")

end
