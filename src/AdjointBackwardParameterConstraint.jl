module AdjointsBackwardObsparam

using CatViews.CatView, Distributions, LineSearches, Optim

export Adjoint, Model, AssimilationResults, initialize_θ!, orbit!, gradient!, cost, assimilate!, obs_mean_var!

include("types_backward_spline_obsparam3.jl")
include("adjoint_backward_spline_obsparam.jl")
include("assimilate_backward_obsparam2.jl")

end
