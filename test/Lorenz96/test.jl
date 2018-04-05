include("../../src/Adjoints.jl")
using Adjoints, DataFrames, PlotlyJS, Gadfly



const N = 5
const true_p = [8., 1.]
const obs_variance = 1.
const obs_iteration = 5
const dt = 0.01
const spinup = 73.
const T = 1.
const steps = Integer(T/dt)
const generation_seed = 0

const t = collect(0.:.01:1.)

const pref = "../../adjoint/data/Lorenz96/N_$N/p_$(join(true_p, "_"))/obsvar_$obs_variance/obsiter_$obs_iteration/dt_$dt/spinup_$spinup/T_$T/seed_$generation_seed/"

const tob = readdlm(pref * "true.tsv")'
const obs = readdlm(pref * "observed.tsv")'
# const tob_df = CSV.read(pref * "true.tsv", nullable=false, delim="\t", header=false)
# const obs_df = CSV.read(pref * "observed.tsv", nullable=false, delim="\t", header=false)

const x0 = randn(5)
const p = randn(2)
const dx0 = [1., 0., 0., 0., 0.]
const dp = zeros(2)
a = Adjoint(dt, steps, obs_variance, obs, x0, p, dx0, dp)
const initial_θ = [randn(5); 2.; 0.5]
const res = minimize!(initial_θ, a)
println(res)
hessian = similar(a.x, 7, 7)
covariance = similar(a.x, 7, 7)
variance = similar(a.x, 7)
stddev = similar(a.x, 7)
covariance!(hessian, covariance, variance, stddev, a)
const true_orbit = scatter3d(;x=tob[1,:],y=tob[2,:], z=tob[3,:], mode="lines", line=attr(color="#1f77b4", width=2))
const assimilated = scatter3d(;x=a.x[1,:], y=a.x[2,:], z=a.x[3,:], mode="lines", line=attr(color="yellow", width=2))
const mask = [all(isfinite.(obs[1:3,_i])) for _i in 1:101]
const obs_pl = scatter3d(;x=obs[1,:][mask], y=obs[2,:][mask], z=obs[3,:][mask], mode="lines", line=attr(color="red", width=5))
PlotlyJS.plot([true_orbit, assimilated])
PlotlyJS.plot([true_orbit, assimilated, obs_pl])

const white_panel = Theme(panel_fill="white")
p_stack = Array{Gadfly.Plot}(0)
for _i in 1:N
    df_tob = DataFrame(t=t, x=tob[_i,:], data_type="true orbit")
    _mask = isfinite.(obs[_i,:])
    df_obs = DataFrame(t=t[_mask], x=obs[_i,:][_mask], data_type="observed")
    df_assim = DataFrame(t=t, x=a.x[_i,:], data_type="assimilated")
    df_all = vcat(df_tob, df_obs, df_assim)
    p_stack = vcat(p_stack, Gadfly.plot(df_all, x="t", y="x", color="data_type", Geom.line, Scale.color_discrete_manual("blue", "red", "green"), Guide.colorkey(title=""), Guide.xlabel("<i>t</i>"), Guide.ylabel("<i>x<sub>$_i</sub></i>"), white_panel))
end
draw(PDF("lorenz.pdf", 24cm, 40cm), vstack(p_stack))
# Gadfly.plot(layer(x=t, y=tob[1,:], Geom.line, Theme(default_color="blue")), layer(x=t, y=a.x[1,:], Geom.line, Theme(default_color="green")), layer(x=t[!obs.na[1,:]], y=obs[1,:][!obs.na[1,:]], Geom.line, Theme(default_color="red")))
