check_bounds_length(initial_lower_bounds, initial_upper_bounds) = length(initial_lower_bounds) == length(initial_upper_bounds) || ArgParse.argparse_error("initial_lower_bounds and initial_upper_bounds must have the same length")
check_bound(initial_lower_bound, initial_upper_bound) = initial_lower_bound < initial_upper_bound || ArgParse.argparse_error("each initial_lower_bound must be lower than initial_upper_bound")
# check_params_length(true_params, initial_lower_bounds) = length(true_params) < length(initial_lower_bounds) || ArgParse.argparse_error("length of initial_lower_bounds and initial_upper_bounds must be greater than the length of true_params")

function check_args(
    settings;
    true_params = nothing,
    initial_lower_bounds = nothing,
    initial_upper_bounds = nothing,
    _...)

    try
        check_bounds_length(initial_lower_bounds, initial_upper_bounds)
        check_bound.(initial_lower_bounds, initial_upper_bounds)
        # check_params_length(true_params, initial_lower_bounds)
    catch err
        isa(err, ArgParseError) || rethrow()
        settings.exc_handler(settings, err)
    end
end
