module AdjointsBackwardObsparam

using CatViews, Distributions, LineSearches, Optim, LinearAlgebra

export Adjoint, Model, AssimilationResults, initialize!, orbit!, orbit_first!, gradient!, cost, assimilate!, obs_mean_var!

include("types_backward_spline_obsparams3.jl")
include("adjoint_backward_spline_obsparam.jl")
include("assimilate_backward_obsparam2.jl")

end
