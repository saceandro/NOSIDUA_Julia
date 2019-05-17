# NOSIDUA_Julia

Julia implementation of NOnlinear dynamical System IDentification with Uncertainty Assessment (NOSIDUA).

usage: twin_experiment [-d DIR] [--number-of-params NUMBER_OF_PARAMS]
                       [--number-of-obs-params NUMBER_OF_OBS_PARAMS]
                       [-p [TRUE_PARAMS...]]
                       [-l INITIAL_LOWER_BOUNDS [INITIAL_LOWER_BOUNDS...]]
                                              [-u INITIAL_UPPER_BOUNDS [INITIAL_UPPER_BOUNDS...]]
                       [--pseudo-obs PSEUDO_OBS [PSEUDO_OBS...]]
                       [--pseudo-obs-var PSEUDO_OBS_VAR [PSEUDO_OBS_VAR...]]
                       [--obs-variance OBS_VARIANCE [OBS_VARIANCE...]]
                       [--obs-iteration OBS_ITERATION] [--dt DT]
                       [--spinup SPINUP] [-t DURATION]
                       [-s GENERATION_SEED] [--trials TRIALS]
                       [--newton-maxiter NEWTON_MAXITER]
                       [--newton-tol NEWTON_TOL]
                       [--regularization-coefficient REGULARIZATION_COEFFICIENT]
                       [--replicates REPLICATES] [--iter ITER]
                       [--time-point TIME_POINT [TIME_POINT...]]
                       [--parameters PARAMETERS [PARAMETERS...]]
                       [--version] [-h]

# Usage

optional arguments:
  -d, --dir DIR         output directory (default: "result")
  --number-of-params NUMBER_OF_PARAMS
                        #parameters (type: Int64, default: 9)
  --number-of-obs-params NUMBER_OF_OBS_PARAMS
                        number of observation parameters (type: Int64,
                        default: 3)
  -p, --true-params [TRUE_PARAMS...]
                        true parameters (type: Float64, default:
                        [-1.60944, -2.30259, -0.510826, -3.91202,
                        -3.68888, -0.693147, 0.223144, -0.356675,
                        -4.60517, 1.0, 1.0, 1.0])
  -l, --initial-lower-bounds INITIAL_LOWER_BOUNDS [INITIAL_LOWER_BOUNDS...]
                        lower bounds for initial state and parameters
                        (type: Float64, default: [-2.30259, -2.30259,
                        -2.30259, -2.30259, -2.30259, -2.30259,
                        -2.30259, -2.30259, -2.30259, -2.30259,
                        -2.99573, -2.30259, -4.60517, -4.60517,
                        -2.30259, 0.0, -2.30259, -5.29832, 0.693147,
                        0.693147, 0.693147])
  -u, --initial-upper-bounds INITIAL_UPPER_BOUNDS [INITIAL_UPPER_BOUNDS...]
                        upper bounds for initial state and parameters
                        (type: Float64, default: [0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.693147, 0.0,
                        -2.99573, -2.99573, 0.0, 1.60944, 0.0,
                        -3.91202, 1.09861, 1.09861, 1.09861])
  --pseudo-obs PSEUDO_OBS [PSEUDO_OBS...]
                        #pseudo observations (type: Int64, default:
                        [0, 0, 0])
  --pseudo-obs-var PSEUDO_OBS_VAR [PSEUDO_OBS_VAR...]
                        variance of pseudo observations (type:
                        Float64, default: [1.0, 1.0, 1.0])
  --obs-variance OBS_VARIANCE [OBS_VARIANCE...]
                        observation variance (type: Float64, default:
                        [0.0001, 0.0001, 0.0001])
  --obs-iteration OBS_ITERATION
                        observation iteration (type: Int64, default:
                        20)
  --dt DT               Δt (type: Float64, default: 0.25)
  --spinup SPINUP       spinup (type: Float64, default: 0.0)
  -t, --duration DURATION
                        assimilation duration (type: Float64, default:
                        35.0)
  -s, --generation-seed GENERATION_SEED
                        seed for orbit generation (type: Int64,
                        default: 0)
  --trials TRIALS       #trials for gradient descent initial value
                        (type: Int64, default: 10)
  --newton-maxiter NEWTON_MAXITER
                        #maxiter for newton's method (type: Int64,
                        default: 1000)
  --newton-tol NEWTON_TOL
                        newton method toralence (type: Float64,
                        default: 1.0e-12)
  --regularization-coefficient REGULARIZATION_COEFFICIENT
                        regularization coefficient (type: Float64,
                        default: 1.0)
  --replicates REPLICATES
                        #replicates (type: Int64, default: 10)
  --iter ITER           #iterations (type: Int64, default: 1)
  --time-point TIME_POINT [TIME_POINT...]
                        time points (type: Float64, default: [0.0,
                        5.0, 10.0, 20.0, 35.0])
  --parameters PARAMETERS [PARAMETERS...]
                        name of the parameters (default: ["log[STAT]",
                        "log[pSTAT]", "log[pSTAT-pSTAT]",
                        "log[npSTAT-npSTAT]", "log[nSTAT1]",
                        "log[nSTAT2]", "log[nSTAT3]", "log[nSTAT4]",
                        "log[nSTAT5]", "logp1", "logp2", "logp3",
                        "logp4", "logu1", "logu2", "logu3", "logu4",
                        "logu5", "Opstat", "Otstat", "Ststat"])
  --version             show version information and exit
  -h, --help            show this help message and exit
