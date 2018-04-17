module Adjoints

using CatViews, NLSolversBase, Optim, Distributions

export Adjoint, Model, minimize!, covariance!, covariance_from_θ0!, orbit_gradient!, numerical_gradient!, numerical_hessian!, assimilate!

include("types.jl")
include("adjoint.jl")
include("assimilate.jl")

end
