module Adjoints

import CatViews.CatView
import Optim: OnceDifferentiable, LBFGS, optimize

export Adjoint, minimize!, covariance!, orbit_gradient!, numerical_gradient!, numerical_hessian!

include("types.jl")
include("adjoint.jl")

end
