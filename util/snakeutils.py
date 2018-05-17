#!/usr/bin/env python

import subprocess, itertools, operator
from functools import reduce
from snakemake.utils import R
from snakemake import shell

def execcommand(cmd,stdout,stderr,input=""):
    with open(stdout, "w") as outf:
        with open (stderr, "w") as errf:
            o,e = subprocess.Popen(cmd, shell=True, stdout=outf, stderr=errf, universal_newlines=True).communicate(input)
            errf.close()
        outf.close()
    return o,e

def makepath(dirdic, config):
    """
    Make base directory path based on fixed dirdic and config.
    e.g.
    dirdic = {'dir': 'result1'}
    config =
    {'true_params':             '8.0 1.0',
     'initial_lower_bounds':    '-10.0 -10.0 -10.0 -10.0 -10.0 0.0 0.0',
     'initial_upper_bounds':    '10.0 10.0 10.0 10.0 10.0 16.0 2.0',
     'spinup':                  73.0,
     'generation_seed':         0,
     'trials':                  50}
        ->
    makepath(dirdic, config) =
    "result1/true_params_8.0_1.0/initial_lower_bounds_-10.0_-10.0_-10.0_-10.0_-10.0_0.0_0.0/initial_upper_bounds_10.0_10.0_10.0_10.0_10.0_16.0_2.0/spinup_73.0/generation_seed_0/trials_50"
    """
    return "/".join(["/".join("{}".format(val) for val in dirdic.values()), "/".join("{}_{}".format(key,str(val).replace(" ", "_")) for (key,val) in config.items())])

def make_dir_path(dirdic):
    return "/".join("{}".format(val) for val in dirdic.values())

def make_plot_wildcard(dirdic, *args):
    return "/".join([make_dir_path(dirdic), "/".join(args)])

def make_param_path(paramdic):
    return "/".join("{}_{}".format(*item) for item in paramdic.items())

def divide_dic(paramsdic):
    return [dict(zip(paramsdic.keys(), item)) for item in itertools.product(*paramsdic.values())]

# def format_dic(basedir, paramdic):
#     return "/".join([basedir, "/".join("{}_{}".format(*item) for item in paramdic.items())])

def format_divide_dic(basedir, paramsdic):
    """
    Make real path directly using the lists-of-parameters dictionary.
    e.g.
    basedir = "result1/true_params_8.0_1.0/initial_lower_bounds_-10.0_-10.0_-10.0_-10.0_-10.0_0.0_0.0/initial_upper_bounds_10.0_10.0_10.0_10.0_10.0_16.0_2.0/spinup_73.0/generation_seed_0/trials_50"
    alldic =
    {'obs_variance':    array([ 1.]),
     'obs_iteration':   array([5]),
     'dt':              array([ 0.01]),
     'duration':        array([ 1.]),
     'replicates':      array([1, 2]),
     'iterations':      array([1, 2])}
        ->
    format_divide_dic(basedir, alldic) =
    ["result1/true_params_8.0_1.0/initial_lower_bounds_-10.0_-10.0_-10.0_-10.0_-10.0_0.0_0.0/initial_upper_bounds_10.0_10.0_10.0_10.0_10.0_16.0_2.0/spinup_73.0/generation_seed_0/trials_50/obs_variance_1.0/obs_iteration_5/dt_0.01/duration_1.0/replicates_1/iterations_1",
     "result1/true_params_8.0_1.0/initial_lower_bounds_-10.0_-10.0_-10.0_-10.0_-10.0_0.0_0.0/initial_upper_bounds_10.0_10.0_10.0_10.0_10.0_16.0_2.0/spinup_73.0/generation_seed_0/trials_50/obs_variance_1.0/obs_iteration_5/dt_0.01/duration_1.0/replicates_1/iterations_2",
     "result1/true_params_8.0_1.0/initial_lower_bounds_-10.0_-10.0_-10.0_-10.0_-10.0_0.0_0.0/initial_upper_bounds_10.0_10.0_10.0_10.0_10.0_16.0_2.0/spinup_73.0/generation_seed_0/trials_50/obs_variance_1.0/obs_iteration_5/dt_0.01/duration_1.0/replicates_2/iterations_1",
     "result1/true_params_8.0_1.0/initial_lower_bounds_-10.0_-10.0_-10.0_-10.0_-10.0_0.0_0.0/initial_upper_bounds_10.0_10.0_10.0_10.0_10.0_16.0_2.0/spinup_73.0/generation_seed_0/trials_50/obs_variance_1.0/obs_iteration_5/dt_0.01/duration_1.0/replicates_2/iterations_2"]
    """
    return ["/".join([basedir, "/".join("{}_{}".format(*i) for i in zip(paramsdic.keys(), item))]) for item in itertools.product(*paramsdic.values())]

def format_divide_dic_file(basedir, paramsdic, filename):
    """
    Make real path directly using the lists-of-parameters dictionary.
    e.g.
    basedir = "result1/true_params_8.0_1.0/initial_lower_bounds_-10.0_-10.0_-10.0_-10.0_-10.0_0.0_0.0/initial_upper_bounds_10.0_10.0_10.0_10.0_10.0_16.0_2.0/spinup_73.0/generation_seed_0/trials_50"
    alldic =
    {'obs_variance':    array([ 1.]),
     'obs_iteration':   array([5]),
     'dt':              array([ 0.01]),
     'duration':        array([ 1.]),
     'replicates':      array([1, 2]),
     'iter':            array([1, 2])}
    filename = "estimates.tsv"
        ->
    format_divide_dic_file(basedir, alldic, filename) =
    ['result1/true_params_8.0_1.0/initial_lower_bounds_-10.0_-10.0_-10.0_-10.0_-10.0_0.0_0.0/initial_upper_bounds_10.0_10.0_10.0_10.0_10.0_16.0_2.0/spinup_73.0/generation_seed_0/trials_50/obs_variance_1.0/obs_iteration_5/dt_0.01/duration_1.0/replicates_1/iter_1/estimates.tsv',
     'result1/true_params_8.0_1.0/initial_lower_bounds_-10.0_-10.0_-10.0_-10.0_-10.0_0.0_0.0/initial_upper_bounds_10.0_10.0_10.0_10.0_10.0_16.0_2.0/spinup_73.0/generation_seed_0/trials_50/obs_variance_1.0/obs_iteration_5/dt_0.01/duration_1.0/replicates_1/iter_2/estimates.tsv',
     'result1/true_params_8.0_1.0/initial_lower_bounds_-10.0_-10.0_-10.0_-10.0_-10.0_0.0_0.0/initial_upper_bounds_10.0_10.0_10.0_10.0_10.0_16.0_2.0/spinup_73.0/generation_seed_0/trials_50/obs_variance_1.0/obs_iteration_5/dt_0.01/duration_1.0/replicates_2/iter_1/estimates.tsv',
     'result1/true_params_8.0_1.0/initial_lower_bounds_-10.0_-10.0_-10.0_-10.0_-10.0_0.0_0.0/initial_upper_bounds_10.0_10.0_10.0_10.0_10.0_16.0_2.0/spinup_73.0/generation_seed_0/trials_50/obs_variance_1.0/obs_iteration_5/dt_0.01/duration_1.0/replicates_2/iter_2/estimates.tsv']
    """
    return ["/".join([basedir, "/".join("{}_{}".format(*i) for i in zip(paramsdic.keys(), item)), filename]) for item in itertools.product(*paramsdic.values())]

def _shell_format_wildcard(paramsdic):
    """
    Make shell parameter given as wildcards using the lists-of-parameters dictionary.
    e.g.
    dir_config_paramsdic =
    {'dir':                     'result1',
     'true_params':             '8.0 1.0',
     'initial_lower_bounds':    '-10.0 -10.0 -10.0 -10.0 -10.0 0.0 0.0',
     'initial_upper_bounds':    '10.0 10.0 10.0 10.0 10.0 16.0 2.0',
     'spinup':                  73.0,
     'generation_seed':         0,
     'trials':                  50,
     'obs_variance':            array([ 1.]),
     'obs_iteration':           array([5]),
     'dt':                      array([ 0.01]),
     'duration':                array([ 1.]),
     'replicates':              array([1, 2])}
        ->
    _shell_format_wildcard(dir_config_paramsdic) =
    "dir={wildcards.dir} true_params={wildcards.true_params} initial_lower_bounds={wildcards.initial_lower_bounds} initial_upper_bounds={wildcards.initial_upper_bounds} spinup={wildcards.spinup} generation_seed={wildcards.generation_seed} trials={wildcards.trials} obs_variance={wildcards.obs_variance} obs_iteration={wildcards.obs_iteration} dt={wildcards.dt} duration={wildcards.duration} replicates={wildcards.replicates}"
    """
    return " ".join("{0}={{wildcards.{0}}}".format(key) for key in paramsdic.keys())

def shell_format_dics_run_wildcard(config_paramsdic, arrayparam, shellfile, out):
    """
    Make array qsub shell command given as wildcards using the lists-of-parameters dictionary.
    e.g.
    dir_config_paramsdic =
    {'dir':                     'result1',
     'true_params':             '8.0 1.0',
     'initial_lower_bounds':    '-10.0 -10.0 -10.0 -10.0 -10.0 0.0 0.0',
     'initial_upper_bounds':    '10.0 10.0 10.0 10.0 10.0 16.0 2.0',
     'spinup':                  73.0,
     'generation_seed':         0,
     'trials':                  50,
     'obs_variance':            array([ 1.]),
     'obs_iteration':           array([5]),
     'dt':                      array([ 0.01]),
     'duration':                array([ 1.]),
     'replicates':              array([1, 2])}
    arrayparam =
    {'iterations': array([1, 2])}
    shellfile = "hoge.sh"
    out = "outfile"
        ->
    shell_format_dics_run_wildcard(dir_config_paramsdic, arrayparam, shellfile, out) =
    "dir={wildcards.dir} true_params={wildcards.true_params} initial_lower_bounds={wildcards.initial_lower_bounds} initial_upper_bounds={wildcards.initial_upper_bounds} spinup={wildcards.spinup} generation_seed={wildcards.generation_seed} trials={wildcards.trials} obs_variance={wildcards.obs_variance} obs_iteration={wildcards.obs_iteration} dt={wildcards.dt} duration={wildcards.duration} replicates={wildcards.replicates} qsub -sync y -t 1:2:1 -tc 500 ./hoge.sh outfile"
    """
    return " ".join([_shell_format_wildcard(config_paramsdic), "qsub -sync y -t 1:{}:1 -tc 500".format(len(next(iter(arrayparam.values())))), "./{}".format(shellfile), out]) # get first arrayparam value non-destructively

def _bigarrayjob(dir_config, paramsdic):
    return " ".join([ " ".join("{}='{}'".format(*item) for item in dir_config.items()), " ".join("{}='{}'".format(key, " ".join(map(str, val))) for (key,val) in paramsdic.items()) ])

def bigarrayjob(dir_config, paramsdic, shellfile, jid, out="", hold_jid=None, sync='n', tc=500):
    """
    Make bigarray qsub shell command using the lists-of-parameters dictionary.
    e.g.
    dir_config =
    {'dir': 'result1',
     'true_params': '8.0 1.0',
     'initial_lower_bounds': '-10.0 -10.0 -10.0 -10.0 -10.0 0.0 0.0',
     'initial_upper_bounds': '10.0 10.0 10.0 10.0 10.0 16.0 2.0',
     'spinup': 73.0,
     'generation_seed': 0,
     'trials': 50}
    alldic =
    {'obs_variance':    array([ 1.]),
     'obs_iteration':   array([5]),
     'dt':              array([ 0.01]),
     'duration':        array([ 1.]),
     'replicates':      array([1, 2]),
     'iter':            array([1, 2])}
    shellfile = "bigarray_qsub_experiment.sh"
    out = "out"
        ->
    bigarrayjob(dir_config, alldic, shellfile, out) =
    "dir='result1' true_params='8.0 1.0' initial_lower_bounds='-10.0 -10.0 -10.0 -10.0 -10.0 0.0 0.0' initial_upper_bounds='10.0 10.0 10.0 10.0 10.0 16.0 2.0' spinup='73.0' generation_seed='0' trials='50' obs_variance='1.0' obs_iteration='5' dt='0.01' duration='1.0' replicates='1 2' iter='1 2' qsub -sync y -t 1:4:1 -tc 500 ./bigarray_qsub_experiment.sh out"
    """
    if hold_jid is None:
        return " ".join([ _bigarrayjob(dir_config, paramsdic), "qsub", "-N {}".format(jid), "-sync {}".format(sync), "-t 1:{}:1".format(reduce(operator.mul, map(len, paramsdic.values()))), "-tc {}".format(tc), "./{}".format(shellfile), out ])
    else:
        return " ".join([ _bigarrayjob(dir_config, paramsdic), "qsub", "-N {}".format(jid), "-hold_jid {}".format(hold_jid), "-sync {}".format(sync), "-t 1:{}:1".format(reduce(operator.mul, map(len, paramsdic.values()))), "-tc {}".format(tc), "./{}".format(shellfile), out ])

def bigarrayjob_noqsub(dir_config, paramsdic, shellfile):
    return " ".join([ _bigarrayjob(dir_config, paramsdic), "./{}".format(shellfile) ])

def format_dic_wildcard(dirdic, config_paramsdic):
    """
    Make wildcarded path directly using the lists-of-parameters dictionary.
    e.g.
    dirdic =
    {'dir': 'result1'}
    config_paramsdic =
    {'true_params':             '8.0 1.0',
     'initial_lower_bounds':    '-10.0 -10.0 -10.0 -10.0 -10.0 0.0 0.0',
     'initial_upper_bounds':    '10.0 10.0 10.0 10.0 10.0 16.0 2.0',
     'spinup':                  73.0,
     'generation_seed':         0,
     'trials':                  50,
     'obs_variance':            array([ 1.]),
     'obs_iteration':           array([5]),
     'dt':                      array([ 0.01]),
     'duration':                array([ 1.]),
     'replicates':              array([1, 2])}
        ->
    format_dic_wildcard(dirdic, config_paramsdic) =
    "{dir}/true_params_{true_params}/initial_lower_bounds_{initial_lower_bounds}/initial_upper_bounds_{initial_upper_bounds}/spinup_{spinup}/generation_seed_{generation_seed}/trials_{trials}/obs_variance_{obs_variance}/obs_iteration_{obs_iteration}/dt_{dt}/duration_{duration}/replicates_{replicates}"
    """
    return "/".join(["/".join("{{{}}}".format(key) for key in dirdic), "/".join("{0}_{{{0}}}".format(key) for key in config_paramsdic.keys())]) # curly braces can be escaped by doubling

def format_dic_with_wildcard_prefix(dirdic, config_paramsdic):
    """
    Make wildcarded path directly using the lists-of-parameters dictionary.
    The prefix "wildcards" is added.
    e.g.
    dirdic =
    {'dir': 'result1'}
    config_paramsdic =
    {'true_params':             '8.0 1.0',
     'initial_lower_bounds':    '-10.0 -10.0 -10.0 -10.0 -10.0 0.0 0.0',
     'initial_upper_bounds':    '10.0 10.0 10.0 10.0 10.0 16.0 2.0',
     'spinup':                  73.0,
     'generation_seed':         0,
     'trials':                  50,
     'obs_variance':            array([ 1.]),
     'obs_iteration':           array([5]),
     'dt':                      array([ 0.01]),
     'duration':                array([ 1.]),
     'replicates':              array([1, 2])}
        ->
    format_dic_with_wildcard_prefix(dirdic, config_paramsdic) =
    "{wildcards.dir}/true_params_{wildcards.true_params}/initial_lower_bounds_{wildcards.initial_lower_bounds}/initial_upper_bounds_{wildcards.initial_upper_bounds}/spinup_{wildcards.spinup}/generation_seed_{wildcards.generation_seed}/trials_{wildcards.trials}/obs_variance_{wildcards.obs_variance}/obs_iteration_{wildcards.obs_iteration}/dt_{wildcards.dt}/duration_{wildcards.duration}/replicates_{wildcards.replicates}"
    """
    return "/".join(["/".join("{{wildcards.{}}}".format(key) for key in dirdic), "/".join("{0}_{{wildcards.{0}}}".format(key) for key in config_paramsdic.keys())]) # curly braces can be escaped by doubling


# API functions
def wildcardparams_expandarrayparams(dirdic, config, paramsdic, arrayparam, filename):
    """
    Output files for the array job rule.
    e.g.
    dirdic =
    {'dir': 'result1'}
    config =
    {'true_params':             '8.0 1.0',
     'initial_lower_bounds':    '-10.0 -10.0 -10.0 -10.0 -10.0 0.0 0.0',
     'initial_upper_bounds':    '10.0 10.0 10.0 10.0 10.0 16.0 2.0',
     'spinup':                  73.0,
     'generation_seed':         0,
     'trials':                  50}
    paramsdic =
    {'obs_variance':            array([ 1.]),
     'obs_iteration':           array([5]),
     'dt':                      array([ 0.01]),
     'duration':                array([ 1.]),
     'replicates':              array([1, 2])}
    arrayparam =
    {'iterations': array([1, 2])}
    filename = "estimates.tsv"
        ->
    wildcardparams_expandarrayparams(dirdic, config, paramsdic, arrayparam, filename) =
    ['{dir}/true_params_{true_params}/initial_lower_bounds_{initial_lower_bounds}/initial_upper_bounds_{initial_upper_bounds}/spinup_{spinup}/generation_seed_{generation_seed}/trials_{trials}/obs_variance_{obs_variance}/obs_iteration_{obs_iteration}/dt_{dt}/duration_{duration}/replicates_{replicates}/iterations_1/estimates.tsv',
     '{dir}/true_params_{true_params}/initial_lower_bounds_{initial_lower_bounds}/initial_upper_bounds_{initial_upper_bounds}/spinup_{spinup}/generation_seed_{generation_seed}/trials_{trials}/obs_variance_{obs_variance}/obs_iteration_{obs_iteration}/dt_{dt}/duration_{duration}/replicates_{replicates}/iterations_2/estimates.tsv']
    """
    return format_divide_dic_file(format_dic_wildcard(dirdic, dict(config, **paramsdic)), arrayparam, filename)

def make_array_shellcommand(dirdic, config, paramsdic, arrayparam, shellfile):
    """
    Make shellcommand for the array job rule.
    e.g.
    dirdic =
    {'dir': 'result1'}
    config =
    {'true_params':             '8.0 1.0',
     'initial_lower_bounds':    '-10.0 -10.0 -10.0 -10.0 -10.0 0.0 0.0',
     'initial_upper_bounds':    '10.0 10.0 10.0 10.0 10.0 16.0 2.0',
     'spinup':                  73.0,
     'generation_seed':         0,
     'trials':                  50}
    paramsdic =
    {'obs_variance':            array([ 1.]),
     'obs_iteration':           array([5]),
     'dt':                      array([ 0.01]),
     'duration':                array([ 1.]),
     'replicates':              array([1, 2])}
    arrayparam =
    {'iterations': array([1, 2])}
        ->
    make_array_shellcommand(dirdic, config, paramsdic, arrayparam, shellfile) =
    "dir={wildcards.dir} true_params={wildcards.true_params} initial_lower_bounds={wildcards.initial_lower_bounds} initial_upper_bounds={wildcards.initial_upper_bounds} spinup={wildcards.spinup} generation_seed={wildcards.generation_seed} trials={wildcards.trials} obs_variance={wildcards.obs_variance} obs_iteration={wildcards.obs_iteration} dt={wildcards.dt} duration={wildcards.duration} replicates={wildcards.replicates} qsub -sync y -t 1:2:1 -tc 500 ./array_qsub_experiment.sh {wildcards.dir}/true_params_{wildcards.true_params}/initial_lower_bounds_{wildcards.initial_lower_bounds}/initial_upper_bounds_{wildcards.initial_upper_bounds}/spinup_{wildcards.spinup}/generation_seed_{wildcards.generation_seed}/trials_{wildcards.trials}/obs_variance_{wildcards.obs_variance}/obs_iteration_{wildcards.obs_iteration}/dt_{wildcards.dt}/duration_{wildcards.duration}/replicates_{wildcards.replicates}"
    """
    config_paramsdic = dict(config, **paramsdic)
    dir_config_paramsdic = dict(dirdic, **config_paramsdic)
    return shell_format_dics_run_wildcard(dir_config_paramsdic, arrayparam, shellfile, format_dic_with_wildcard_prefix(dirdic, config_paramsdic))

def make_all_output_filenames(dirdic, config, paramsdic, arrayparam, filename):
    """
    Make all output filenames.
    e.g.
    dirdic =
    {'dir': 'result1'}
    config =
    {'true_params':             '8.0 1.0',
     'initial_lower_bounds':    '-10.0 -10.0 -10.0 -10.0 -10.0 0.0 0.0',
     'initial_upper_bounds':    '10.0 10.0 10.0 10.0 10.0 16.0 2.0',
     'spinup':                  73.0,
     'generation_seed':         0,
     'trials':                  50}
    paramsdic =
    {'obs_variance':            array([ 1.]),
     'obs_iteration':           array([5]),
     'dt':                      array([ 0.01]),
     'duration':                array([ 1.]),
     'replicates':              array([1, 2])}
    arrayparam =
    {'iterations': array([1, 2])}
    filename = "estimates.tsv"
        ->
    make_all_output_filenames(dirdic, config, paramsdic, arrayparam, filename) =
    ['result1/true_params_8.0_1.0/initial_lower_bounds_-10.0_-10.0_-10.0_-10.0_-10.0_0.0_0.0/initial_upper_bounds_10.0_10.0_10.0_10.0_10.0_16.0_2.0/spinup_73.0/generation_seed_0/trials_50/obs_variance_1.0/obs_iteration_5/dt_0.01/duration_1.0/replicates_1/iter_1/estimates.tsv',
     'result1/true_params_8.0_1.0/initial_lower_bounds_-10.0_-10.0_-10.0_-10.0_-10.0_0.0_0.0/initial_upper_bounds_10.0_10.0_10.0_10.0_10.0_16.0_2.0/spinup_73.0/generation_seed_0/trials_50/obs_variance_1.0/obs_iteration_5/dt_0.01/duration_1.0/replicates_1/iter_2/estimates.tsv',
     'result1/true_params_8.0_1.0/initial_lower_bounds_-10.0_-10.0_-10.0_-10.0_-10.0_0.0_0.0/initial_upper_bounds_10.0_10.0_10.0_10.0_10.0_16.0_2.0/spinup_73.0/generation_seed_0/trials_50/obs_variance_1.0/obs_iteration_5/dt_0.01/duration_1.0/replicates_2/iter_1/estimates.tsv',
     'result1/true_params_8.0_1.0/initial_lower_bounds_-10.0_-10.0_-10.0_-10.0_-10.0_0.0_0.0/initial_upper_bounds_10.0_10.0_10.0_10.0_10.0_16.0_2.0/spinup_73.0/generation_seed_0/trials_50/obs_variance_1.0/obs_iteration_5/dt_0.01/duration_1.0/replicates_2/iter_2/estimates.tsv']
    """
    return format_divide_dic_file(makepath(dirdic, config), dict(paramsdic, **arrayparam), filename)

def make_all_output_filenames_all(dirdic, config, paramsdics, arrayparam, filename):
    output_files = []
    for paramsdic in paramsdics.values():
        output_files += make_all_output_filenames(dirdic, config, paramsdic, arrayparam, filename)
    return output_files

def bigarrayjob_run(dirdic, config, paramsdic, arrayparam, shellfile, jid, out="", hold_jid=None, sync='n', tc=500):
    """
    Make bigarray qsub shell command using the lists-of-parameters dictionary.
    e.g.
    dirdic =
    {'dir': 'result1'}
    config =
    {'true_params':             '8.0 1.0',
     'initial_lower_bounds':    '-10.0 -10.0 -10.0 -10.0 -10.0 0.0 0.0',
     'initial_upper_bounds':    '10.0 10.0 10.0 10.0 10.0 16.0 2.0',
     'spinup':                  73.0,
     'generation_seed':         0,
     'trials':                  50}
    paramsdic =
    {'obs_variance':            array([ 1.]),
     'obs_iteration':           array([5]),
     'dt':                      array([ 0.01]),
     'duration':                array([ 1.]),
     'replicates':              array([1, 2])}
    arrayparam =
    {'iterations': array([1, 2])}
    shellfile = "bigarray_qsub_experiment.sh"
        ->
    bigarrayjob_run(dirdic, config, paramsdic, arrayparam, shellfile) =
    "dir='result1' true_params='8.0 1.0' initial_lower_bounds='-10.0 -10.0 -10.0 -10.0 -10.0 0.0 0.0' initial_upper_bounds='10.0 10.0 10.0 10.0 10.0 16.0 2.0' spinup='73.0' generation_seed='0' trials='50' obs_variance='1.0' obs_iteration='5' dt='0.01' duration='1.0' replicates='1 2' iter='1 2' qsub -sync y -t 1:4:1 -tc 500 ./bigarray_qsub_experiment.sh "
    """
    return bigarrayjob(dict(dirdic, **config), dict(paramsdic, **arrayparam), shellfile, jid, out=out, hold_jid=hold_jid, sync=sync, tc=tc)

def bigarrayjob_run_all_hqw(dirdic, config, paramsdics, arrayparam, shellfile, tc=500):
    test_cases = list(paramsdics.keys())
    paramsdics_list = list(paramsdics.values())
    shell(bigarrayjob_run(dirdic, config, paramsdics_list[0], arrayparam, shellfile, test_cases[0], tc=tc))
    for i in range(1,len(paramsdics_list)-1):
        shell(bigarrayjob_run(dirdic, config, paramsdics_list[i], arrayparam, shellfile, test_cases[i], hold_jid=test_cases[i-1], tc=tc))
    shell(bigarrayjob_run(dirdic, config, paramsdics_list[-1], arrayparam, shellfile, test_cases[-1], hold_jid=test_cases[-2], sync='y', tc=tc))
    # return [bigarrayjob_run(dirdic, config, paramsdics_list[0], arrayparam, shellfile, test_cases[0], tc=tc)]\
    #      + [bigarrayjob_run(dirdic, config, paramsdics_list[i], arrayparam, shellfile, test_cases[i], hold_jid=test_cases[i-1], tc=tc) for i in range(1,len(paramsdics_list)-1)]\
    #      + [bigarrayjob_run(dirdic, config, paramsdics_list[-1], arrayparam, shellfile, test_cases[-1], hold_jid=test_cases[-2], sync='y', tc=tc)]

def plotdic(paramsdics):
    return {meta_key: {key: val for (key,val) in paramsdics[meta_key].items() if key is not meta_key} for meta_key in paramsdics.keys()}

def totalnames(plotdir, config, paramsdic, variables):
    return list(itertools.chain.from_iterable(format_divide_dic_file(makepath( dict(plotdir, **{"testcase": testcase}), config), paramdic, item) for (testcase, paramdic) in plotdic(paramsdic).items() for item in variables) )

def plotnames(plotdir, config, paramsdic, variables, kindofplot):
    return list(itertools.chain.from_iterable(format_divide_dic_file(makepath( dict(plotdir, **{"testcase": testcase}), config), paramdic, "/".join([kindofplot, item + ".pdf"])) for (testcase, paramdic) in plotdic(paramsdic).items() for item in variables) )

def total(dirdic, plotdir, config, paramsdic, arrayparam, variables, resultfilename):
    for (testcase, paramdic) in plotdic(paramsdic).items():
        plotbase = format_divide_dic(makepath( dict(plotdir, **{"testcase": testcase}), config), paramdic)[0]
        totalfiles = ["/".join([plotbase, item]) for item in variables]
        totalfiles_stream = [open(totalfile, "w") for totalfile in totalfiles]
        for totalfile_s in totalfiles_stream:
            totalfile_s.write("%s\tdiff\tCI\n" % testcase)
        for paramdic_all in divide_dic(paramsdic[testcase]):
            resultbase = makepath( dirdic, dict(config, **paramdic_all) )
            for _iter in divide_dic(arrayparam):
                resultfile = "/".join([resultbase, make_param_path(_iter), resultfilename])
                with open(resultfile, "r") as resultf:
                    for i in range(len(variables)):
                        totalfiles_stream[i].write("%f\t%s" % (paramdic_all[testcase], resultf.readline()))
                    resultf.close()
        for item in totalfiles_stream:
            item.close()

def boxplot(input, output, x, y, x_log_scale_base=None, y_log_scale_base=None, remove_na=False):
    command = """
    library(ggplot2)
    library(scales)
    d <- read.delim("{input}", header=T)
    """
    if remove_na:
        command += """
        d <- d[complete.cases(d),]
        """
    command += """
    g <- ggplot(d, aes(x={x}, y={y}, group={x}))
    g <- g + geom_boxplot()
    # g <- g + geom_point()
    """
    # if ((x_log_scale_base=="10") and (y_log_scale_base=="10")):
    #     command += """
    #     g <- g + coord_trans(x = "log10", y = "log10")
    #     """
    if x_log_scale_base is not None:
        command += """
        g <- g + scale_x_continuous(
            trans = 'log{x_log_scale_base}',
            breaks = pretty_breaks()
            # labels = trans_format('log{x_log_scale_base}', math_format({x_log_scale_base}^.x))
            )
        """
    if y_log_scale_base is not None:
        command += """
        g <- g + scale_y_continuous(
            trans = 'log{y_log_scale_base}',
            breaks = pretty_breaks()
            # labels = trans_format('log{y_log_scale_base}', math_format({y_log_scale_base}^.x))
            )
        """
    command += """
    ggsave(file="{output}", plot=g)
    """
    command = command.format(input=input, output=output, x=x, y=y, x_log_scale_base=x_log_scale_base, y_log_scale_base=y_log_scale_base)
    print(command)
    R(command)

def summmarized_plot(input, output, x, y1, y2, y1_op="sd", y2_op="mean", x_log_scale_base=None, y_log_scale_base=None, remove_na=False):
    command = """
    library(ggplot2)
    library(scales)
    library(plyr)
    d <- read.delim("{input}", header=T)
    """
    if remove_na:
        command += """
        d <- d[complete.cases(d),]
        """
    command += """
    d_stat <- ddply(d, .({x}), summarize, "{y1_op}_{y1}"={y1_op}({y1}), "{y2_op}_{y2}"={y2_op}({y2}))
    d1 <- data.frame(d_stat${x}, d_stat${y1_op}_{y1}, "{y1_op}_{y1}")
    colnames(d1) <- c("{x}", "{y1}", "group")
    d2 <- data.frame(d_stat${x}, d_stat${y2_op}_{y2}, "{y2_op}_{y2}")
    colnames(d2) <- c("{x}", "{y1}", "group")
    d_merged <- rbind(d1, d2)
    g <- ggplot(d_merged, aes(x={x}, y={y1}, group=group, color=group))
    g <- g + geom_point()
    g <- g + geom_line()
    """
    # if ((x_log_scale_base=="10") and (y_log_scale_base=="10")):
    #     command += """
    #     g <- g + coord_trans(x = "log10", y = "log10")
    #     """
    if x_log_scale_base is not None:
        command += """
        g <- g + scale_x_continuous(
            trans = 'log{x_log_scale_base}',
            breaks = pretty_breaks()
            # breaks = trans_breaks('log{x_log_scale_base}', function(x) {x_log_scale_base}^(x/2)),
            # labels = trans_format('log{x_log_scale_base}', math_format({x_log_scale_base}^.x))
            )
        """
    if y_log_scale_base is not None:
        command += """
        g <- g + scale_y_continuous(
            trans = 'log{y_log_scale_base}',
            breaks = pretty_breaks()
            # breaks = trans_breaks('log{y_log_scale_base}', function(x) {y_log_scale_base}^(x/2)),
            # labels = trans_format('log{y_log_scale_base}', math_format({y_log_scale_base}^.x))
            )
        """
    command += """
    g <- g + scale_color_hue(name="", labels=c({y1_op}_{y1}="{y1_op}({y1})", {y2_op}_{y2}="{y2_op}({y2})"))
    ggsave(file="{output}", plot=g)
    """
    command = command.format(input=input, output=output, x=x, y1=y1, y2=y2, y1_op=y1_op, y2_op=y2_op, x_log_scale_base=x_log_scale_base, y_log_scale_base=y_log_scale_base)
    print(command)
    R(command)
