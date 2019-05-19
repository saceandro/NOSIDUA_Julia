# NOSIDUA_Julia

Julia implementation of NOnlinear dynamical System IDentification with Uncertainty Assessment (NOSIDUA).
Command line options and more functions are provided compared to python version.

# Requirements
Julia 0.6.4

Packages:
CatViews
Distributions
LineSearches
Optim
ArgParse

# Usage
Please see NOSIDUA_Julia_application.pdf for the following insulin signaling-dependent gene expression model (michaelis_foldchange_positive_backward_parameter).

```
$ cd test/michaelis_foldchange_positive_backward_parameter
$ julia michaelis_obs_m_test.jl

$ julia michaelis_obs_m_test.jl --help
usage: michaelis_obs_m [-d DIR] [-p [TRUE_PARAMS...]]
                       [-l INITIAL_LOWER_BOUNDS [INITIAL_LOWER_BOUNDS...]]
                                              [-u INITIAL_UPPER_BOUNDS [INITIAL_UPPER_BOUNDS...]]
                       [--pseudo-obs PSEUDO_OBS [PSEUDO_OBS...]]
                       [--pseudo-obs-var PSEUDO_OBS_VAR [PSEUDO_OBS_VAR...]]
                       [--obs-variance OBS_VARIANCE]
                       [--obs-iteration OBS_ITERATION] [--dt DT]
                       [--spinup SPINUP] [-t DURATION]
                       [--trials TRIALS]
                       [--newton-maxiter NEWTON_MAXITER]
                       [--newton-tol NEWTON_TOL]
                       [--regularization-coefficient REGULARIZATION_COEFFICIENT]
                       [--replicates REPLICATES] [--iter ITER]
                       [--x0 X0 [X0...]] [--version] [-h]

Adjoint method

optional arguments:
  -d, --dir DIR         output directory (default: "result")
  -p, --true-params [TRUE_PARAMS...]
                        true parameters (type: Float64, default:
                        [-1.20397, 0.0, 1.60944, -1.89712])
  -l, --initial-lower-bounds INITIAL_LOWER_BOUNDS [INITIAL_LOWER_BOUNDS...]
                        lower bounds for initial state and parameters
                        (type: Float64, default: [-2.30259, -2.30259,
                        0.0, -2.30259])
  -u, --initial-upper-bounds INITIAL_UPPER_BOUNDS [INITIAL_UPPER_BOUNDS...]
                        upper bounds for initial state and parameters
                        (type: Float64, default: [0.0, 2.30259,
                        2.30259, 0.0])
  --pseudo-obs PSEUDO_OBS [PSEUDO_OBS...]
                        #pseudo observations (type: Int64, default:
                        [0, 0])
  --pseudo-obs-var PSEUDO_OBS_VAR [PSEUDO_OBS_VAR...]
                        variance of pseudo observations (type:
                        Float64, default: [1.0, 1.0])
  --obs-variance OBS_VARIANCE
                        observation variance (type: Float64, default:
                        0.001)
  --obs-iteration OBS_ITERATION
                        observation iteration (type: Int64, default:
                        5)
  --dt DT               Δt (type: Float64, default: 0.1)
  --spinup SPINUP       spinup (type: Float64, default: 0.0)
  -t, --duration DURATION
                        assimilation duration (type: Float64, default:
                        240.0)
  --trials TRIALS       #trials for gradient descent initial value
                        (type: Int64, default: 20)
  --newton-maxiter NEWTON_MAXITER
                        #maxiter for newton's method (type: Int64,
                        default: 100)
  --newton-tol NEWTON_TOL
                        newton method toralence (type: Float64,
                        default: 1.0e-8)
  --regularization-coefficient REGULARIZATION_COEFFICIENT
                        regularization coefficient (type: Float64,
                        default: 1.0)
  --replicates REPLICATES
                        #replicates (type: Int64, default: 100)
  --iter ITER           #iterations (type: Int64, default: 1)
  --x0 X0 [X0...]       initial x (type: Float64, default: [0.0, 0.0])
  --version             show version information and exit
  -h, --help            show this help message and exit


```


## Extract from output
```
mincost:        -165590.67073874315
p:              [-1.22553, -0.0158287, 1.61344, -1.89201]
ans:            [-1.20397, 0.0, 1.60944, -1.89712]
diff:           [-0.0215538, -0.0158287, 0.00399829, 0.00510693]
0.013758449087601364
precision:      [8.75505e5 1.31006e5 4.34241e6 6.53755e5; 1.32353e5 8.51591e5 1.45594e6 2.06237e6; 4.34429e6 1.45344e6 2.24773e7 4.98383e6; 6.56896e5 2.06015e6 4.98675e6 5.29142e6]
CI:             [0.0455814, 0.0282653, 0.00879531, 0.00848429]
obs variance:   [NaN, 0.00100802]
```

## Description of the output
```
p:            estimated parameters
ans:          true parameters
diff:         difference between estimated parameters and true parameters
precision:    precisoin matrix of the estimated parameters
CI:           confidence interval of the estimated parameters
obs variance: estimated variance of observation (first element is NaN because no observation is provided for the first element)
```
