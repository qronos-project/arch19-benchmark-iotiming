#!/usr/bin/python2
# We need to use python2 because jinja2 does not offer stable support for python3 yet :-(
from __future__ import print_function
from jinja2 import Environment, FileSystemLoader, StrictUndefined
import numpy as np
from scipy.linalg import expm
import sys
import os
import shutil
import copy
import logging
import pickle
from pprint import PrettyPrinter
import math
from collections import OrderedDict

hypy_available = False
try:
    from hybridpy import hypy
    hypy_available = True
except ImportError as e:
    logging.warning("hypy not found, some functions will be unavailable")
    if not "--ignore-hypy" in sys.argv:
        logging.fatal("hypy not found. Use --ignore-hypy to ignore that.")
        raise e
        sys.exit(1)

os.chdir(os.path.dirname(sys.argv[0]))
if not os.path.isfile("template_system.xml.j2"):
    print("must be run from within the script directory as ./template.py")
    sys.exit(1)



def blkrepeat(M, repetitions):
    '''
    block-diagonal repetition of matrix M.
    M is converted to np.array.

    blkrepeat(M,3)= blkdiag(M,M,M) = [M,
                                        M,
                                           M]
    '''
    return np.kron(np.eye(repetitions), np.asarray(M))

def blockmatrix(M, blocklengths):
    '''
    build a square block-matrix like np.block, where
    0 is replaced by zeroes(...) of appropriate dimension.

    blocklengths is an array of the length of each block. The matrices on the diagonal must be square.

    Example:
    blockmatrix([[A, B], [0, C]], [a,b]) = np.block([[A, B], [zeroes(b,a), C]]).
    with matrices A,B,C of shape (a,a), (a,b), and (b,b) respectively.
    '''
    assert isinstance(M, list)
    assert len(M) == len(blocklengths)
    for i in M:
        assert isinstance(i, list), "M must be a list of lists of matrices"
        assert len(i) == len(blocklengths), "each row of M must have as many entries as there are blocks"

    output = np.zeros((sum(blocklengths), sum(blocklengths)));
    for i in range(len(blocklengths)):
        for j in range(len(blocklengths)):
            block_value = M[i][j]
            if type(block_value) == type(0) and block_value == 0:
                # replace integer 0 with np.zeros(...) of appropriate dimension
                block_value = np.zeros((blocklengths[i], blocklengths[j]))
            output[sum(blocklengths[0:i]):sum(blocklengths[0:i+1]), sum(blocklengths[0:j]):sum(blocklengths[0:j+1])] = block_value
    return output

class System(object):
    def __init__(self):
        self.A_p = None
        self.B_p = None
        self.C_p = None

        self.x_p_0_min = None
        self.x_p_0_max = None
        
        # NOTE: Disturbance is not supported yet. If support is added, please make sure to also add it in the to_latex() function.

        self.A_d = None
        self.B_d = None
        self.C_d = None

        self.T = None
        self.delta_t_u_max = None
        self.delta_t_u_min = None
        self.delta_t_y_max = None
        self.delta_t_y_min = None

        # enable to add the global time t as a state for debugging
        # If enabled, the reachability computation cannot terminate, but its intermediate result can be useful to check the behavior over time and the liveness of the automaton (t must keep growing, not get stuck).
        self.global_time = False

        # SpaceEx: simulation only? (deterministic x0)
        self.spaceex_only_simulate = False
        
        # Normally, the automaton uses nondeterministic transitions in the semantics of Henzinger (same as SpaceEx). We need that because the sampling transition *does not necessarily have to* happen at the first possible time, it may also happen at some other time in the permitted interval [delta_t_min, delta_t_max].
        # This means that a simulator would find a "random tree" of possible solutions, however pysim does not support that and always takes the first possible transition ("urgent semantics").
        # As a workaround, this option can be enabled, so that the automaton is written in urgent semantics, and the randomness is modeled by including a (very weak, not truly random!) "pseudo-random" number generator (PRNG) in the automaton.
        # If enabled, this means that for zero disturbance, the simulation result only depends on the initial state and the simulator does not have to understand the semantics of nondeterministic, non-urgent transitions
        # This option does not make much sense for analysis, even though -- assuming the analysis tool uses urgent semantics -- it should in theory yield correct results with significantly less efficiency. (Except if the analysis tool is clever enough to use the weakness (non-randomness) of the pseudorandom number generator to exclude some traces that the RNG will never reach.)
        self.use_urgent_semantics_and_pseudorandom_sequence = False

        # model transformation to C_d=I, C_p=I
        self.transform_states_as_outputs=False

        # spaceEx configuration parameters from .cfg file
        # maximum iterations
        self.spaceex_iterations = 2000
        # maximum iterations for "reachability over global time" plot (None: same as spaceex_iterations)
        self.spaceex_iterations_for_global_time = None
        # Scenario (overapproximation) - ignored for simulation
        self.spaceex_scenario="stc"
        # approximation directions
        self.spaceex_directions = "oct"
        # sampling time for time elapse
        self.spaceex_sampling_time=1e-3
        # aggregation
        self.spaceex_set_aggregation="none"
        # clustering percentage
        self.spaceex_clustering_percent=100
        
        # results of analysis and simulation
        self.results = {}


    def _check_and_update_dimensions(self):
        '''
        determine whether object is np.array and not np.matrix

        (Note that array has slightly different semantics than matrix!)
        We ensure that objects are not numpy matrices, because they behave slightly differently - see https://www.numpy.org/devdocs/user/numpy-for-matlab-users.html#array-or-matrix-which-should-i-use
        This is out of an abundance of caution.
        '''

        def isarray(a):
            return isinstance(a, np.ndarray) and not isinstance(a, type(np.matrix([])))

        # This is one of the few parts where I hate Python and love Java: Checking types.
        for m in [self.A_p, self.B_p, self.C_p, self.A_d, self.B_d, self.C_d]:
            assert isarray(m), "A,B,C must be of type numpy.array, but one is of type {}. System: {}".format(type(m), self)
        for a in [self.delta_t_u_min, self.delta_t_u_max, self.delta_t_y_min, self.delta_t_y_max]:
            assert isarray(m), "delta_t_... must be of type numpy.array"
        for a in [self.x_p_0_min, self.x_p_0_max]:
            assert isarray(m), "x_p_0_... must be of type numpy.array"

        self.n_p = self.A_p.shape[0]
        assert self.n_p >= 1, "system must have at least one continuous state"
        self.n_d = self.A_d.shape[0]
        self.m = self.B_p.shape[1]
        self.p = self.C_p.shape[0]
        assert self.A_p.shape == (self.n_p, self.n_p)
        assert self.B_p.shape == (self.n_p, self.m)
        assert self.C_p.shape == (self.p, self.n_p)
        assert self.A_d.shape == (self.n_d, self.n_d)
        assert self.B_d.shape == (self.n_d, self.p)
        assert self.C_d.shape == (self.m, self.n_d)
        assert self.x_p_0_min.shape == self.x_p_0_max.shape
        assert self.x_p_0_max.shape == (self.n_p, )

        assert self.delta_t_u_min.shape == self.delta_t_u_max.shape
        assert self.delta_t_u_max.shape == (self.m, )
        assert self.delta_t_y_min.shape == self.delta_t_y_max.shape
        assert self.delta_t_y_max.shape == (self.p, )
        assert all(self.delta_t_u_max >= self.delta_t_u_min), "delta_t_u_max={} is not >= delta_t_u_min={}".format(self.delta_t_u_max, self.delta_t_u_min)
        assert all(self.delta_t_y_max >= self.delta_t_y_min), "delta_t_y_max={} is not >= delta_t_y_min={}".format(self.delta_t_y_max, self.delta_t_y_min)
        assert all(self.x_p_0_max >= self.x_p_0_min)

    def __repr__(self):
        ret = "System("
        for (key,val) in sorted(self.__dict__.iteritems()):
            multiline_variable_indentation = "\n" + " " * (len(key)+6)
            ret += "\n   {} = {}".format(key, multiline_variable_indentation.join(repr(val).split("\n")))
        ret += ")"
        # TODO: unfortunately, the output of this is not directly a valid python expression for getting the system object.
        return ret
    
    def to_latex(self):
        def number_to_latex(number):
            s=str(number)
            if s.endswith(".0"):
                s=s[:-2]
            return r"\num{" + s + "}"
        def numpy_to_latex(matrix):
            if isinstance(matrix, str):
                return matrix
            if isinstance(matrix, (float, int)):
                return number_to_latex(matrix)
            if matrix.shape == (1,):
                # 1d scalar
                return number_to_latex(matrix[0])
            elif matrix.shape ==  (1,1):
                # 2d scalar
                return number_to_latex(matrix[0,0])
            elif len(matrix.shape) == 1:
                return r"\mat{" + r"\\".join([number_to_latex(row) for row in matrix]) + r"} "
            else:
                assert len(matrix.shape) == 2
                return r"\mat{" + r"\\".join([" & ".join([number_to_latex(col) for col in row]) for row in matrix]) + r"} "
        def numpy_interval_to_latex(minv, maxv):
            """
            format multidimensional interval [minv, maxv] as LaTeX
            @param minv: np.ndarray
            @param maxv: np.ndarray
            """
            dimension = minv.shape[0]
            FORMAT = r"[{low}; {high}]"
            if np.all(minv == minv[0]) and np.all(maxv == maxv[0]):
                s = FORMAT.format(low=minv[0], high=maxv[0])
                if dimension > 1:
                    s += "^{}".format(dimension)
                return s
            return r" \times ".join([FORMAT.format(low=minv[i], high=maxv[i]) for i in range(dimension)])
        variables = OrderedDict()
        variables["A_p"] = self.A_p
        variables["B_p"] = self.B_p
        variables["C_p"] = self.C_p
        variables["A_d"] = self.A_d
        variables["B_d"] = self.B_d
        variables["C_d"] = self.C_d
        variables["T"] = self.T
        variables["X_{\subsPlant,0}"] = numpy_interval_to_latex(self.x_p_0_min, self.x_p_0_max)
        variables[r"\underline {\Delta t}_u"] = self.delta_t_u_min
        variables[r"\overline {\Delta t}_u"] = self.delta_t_u_max
        variables[r"\underline {\Delta t}_y"] = self.delta_t_y_min
        variables[r"\overline {\Delta t}_y"] = self.delta_t_y_max
        variables[r"n_{dist}"] = 0 # not supported yet
        return r"\begin{align}" + "\\\\ \n".join(["{name} &= {value}".format(name=name, value=numpy_to_latex(value)) for (name, value) in variables.iteritems()]) + "\n" + r"\end{align}"

    def increase_dimension(self, factor):
        '''
        increase the state dimension by repeating the system n times.
        The resulting system has a block-diagonal structure.
        '''
        self._check_and_update_dimensions()
        assert factor >= 1
        assert isinstance(factor, int)

        self.A_p = blkrepeat(self.A_p, factor)
        self.B_p = blkrepeat(self.B_p, factor)
        self.C_p = blkrepeat(self.C_p, factor)
        self.A_d = blkrepeat(self.A_d, factor)
        self.B_d = blkrepeat(self.B_d, factor)
        self.C_d = blkrepeat(self.C_d, factor)
        self.x_p_0_max = np.tile(self.x_p_0_max, factor)
        self.x_p_0_min = np.tile(self.x_p_0_min, factor)
        self.delta_t_u_max = np.tile(self.delta_t_u_max, factor)
        self.delta_t_u_min = np.tile(self.delta_t_u_min, factor)
        self.delta_t_y_max = np.tile(self.delta_t_y_max, factor)
        self.delta_t_y_min = np.tile(self.delta_t_y_min, factor)
        self._check_and_update_dimensions()

    def nominal_case_stability(self):
        '''
        Check stability for delta_t_...=0
        @return True if stable, False if unknown or unstable.
        '''
        self._check_and_update_dimensions()
        # rewrite system as linear impulsive system, similar to [Gaukler et al. HSCC 2017].
        # Note that it does not matter for stability when the controller is computed,
        # as long as it is between the previous updates of u,y and the next update of u,y.
        # This is because the controller does not directly affect the physical plant.
        # Therefore, if there are no delays, we can move the controller computation to
        # "just before" t=kT, immediately before the update of u and y.
        # The new system now is continuous-time between kT and (k+1)T, and jumps at kT.
        #
        # new total state: x_total = [x_p; x_d; u; y]
        # dimensions of state components:
        blocklengths = [self.n_p, self.n_d, self.m, self.p]

        # continuous dynamics (except at t=kT):
        # x_total' = A_cont_total * x_total
        A_cont_total = blockmatrix([[self.A_p, 0, self.B_p, 0],
                                    [0,        0, 0,        0],
                                    [0,        0, 0,        0],
                                    [0,        0, 0,        0]],
                                   blocklengths)
        # at t=kT   (we moved the controller computation to t=kT, without loss of generality, see note above)
        # compute controller, then update u (currently computed value) and y (for next computation)
        A_discrete_total = blockmatrix([[np.eye(self.n_p),        0,        0,   0],
                                        [0,        self.A_d,        0,   self.B_d],
                                        [0,        self.C_d, 0,   0],
                                        [self.C_p, 0,        0,   0]],
                                       blocklengths)
        A_total = expm(A_cont_total*self.T).dot(A_discrete_total)
        worst_eigenvalue = max(abs(np.linalg.eigvals(A_total)))
        accuracy = 1e-5
        if worst_eigenvalue < 1 - accuracy:
            return 'stable'
        elif worst_eigenvalue > 1 + accuracy:
            return 'unstable'
        else:
            # numerically at the edge of "stable" and "unstable",
            # for example: sine oscillator, integrator ( = marginally stable) or double integrator ( = unstable!)
            return 'borderline unstable'

    def is_nominal_timing(self):
        '''
        are all delta_t == 0?
        '''
        for delta_t in [self.delta_t_u_min, self.delta_t_u_max, self.delta_t_y_min, self.delta_t_y_max]:
            if any(delta_t != 0):
                return False
        return True
    
    def is_fixed_timing(self):
        '''
        are all delta_t_min == delta_t_max?
        '''
        return all(self.delta_t_u_min == self.delta_t_u_max) and all(self.delta_t_y_min == self.delta_t_y_max)

    def to_spaceex(self):
        '''
        convert to SpaceEx format
        @return {'system': system in SpaceEx XML, 'config': configuration for spaceEx web frontend}
        '''
        self._check_and_update_dimensions()
        system=copy.deepcopy(self)

        if system.spaceex_only_simulate:
            system.x_p_0_min = system.x_p_0_max
            system.spaceex_scenario = "simu"

        context = { 'sys': system, 'set_of_io_dimensions': set([system.n_p, system.n_d])}
        output = {}
        env = Environment(autoescape=False, loader=FileSystemLoader('./'), trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined)
        output['system'] = env.get_template('template_system.xml.j2').render(context)
        output['config'] = env.get_template('template_config.cfg.j2').render(context)
        return output

    def to_spaceex_files(self, path_and_prefix):
        '''
        write to files in SpaceEx format
        @param path_and_prefix output path and prefix, e.g. "./output/spaceex-example-", which will be prepended to all filenames.
        @return filename of SpaceEx XML file
        '''
        output = self.to_spaceex()
        with open(path_and_prefix + ".spaceex.xml",'w') as f:
            f.write(output['system'])
        with open(path_and_prefix + '.spaceex.cfg','w') as f:
            f.write(output['config'])
        with open(path_and_prefix + '.info.txt','w') as f:
            f.write('autogenerated. do not edit.\n To use in SpaceEx VM server: First load the .xml as Model File, then .cfg as Configuration File, then start.\n\n Model parameters:\n\n' + repr(self))
        with open(path_and_prefix + '.tex','w') as f:
            f.write(self.to_latex())
        return path_and_prefix + ".spaceex.xml"
    
    def run_analysis(self, name, outdir="./output/"):
        '''
        Analyse system with SpaceEx (practical stability via fixpoint computation) and pysim (Simulation).
        This also generates plots and saves the model files used for analysis.
        
        @param name System name for plots and other output files
        @param outdir Output directory for plots and other output files
        '''
        print("")
        print("====================")
        print("Analysing System: " + name)
        sys.stdout.flush()
        # NOTE: Hyst's SpaceEx-conversion doesn't like dashes in filenames. (Bugfix submitted: https://github.com/verivital/hyst/pull/43 )
        model_file = system.to_spaceex_files(outdir+name)
        
        system.global_time = True
        model_file_global_time = system.to_spaceex_files(outdir + name + "__reachability_with_time_")
        
        system.spaceex_only_simulate=True
        system.to_spaceex_files(outdir + name + "__simulation_with_time_")
        
        system.spaceex_only_simulate=False
        system.use_urgent_semantics_and_pseudorandom_sequence = True
        model_file_pysim = system.to_spaceex_files(outdir + name + "__for_pysim__.")
        
        # Nominal case?
        if system.is_nominal_timing():
            system.results['stability_eigenvalues'] = system.nominal_case_stability()
        else:
            # not the nominal case, cannot test stability via eigenvalues of nominal case
            system.results['stability_eigenvalues'] = 'N/A'
        
        if not hypy_available:
            logging.warning("hypy is unavailable. Not running SpaceEx and other tools.")
            return
        
        # PySim config
        # number of random initial states
        # (mostly for illustration: usually these are not near maximum x_p, and therefore the resulting trajectories are not useful to determine the interval bounds)
        pysim_rand_points=50
        if '--fast' in sys.argv:
            pysim_rand_points=3
        # additionally use initial states at corners?
        # Disabled if the number of states is large
        pysim_initstate_corners = True
        if (system.n_p + system.n_d + system.m + system.p) > 8 or '--fast' in sys.argv:
            pysim_initstate_corners = False
        pysim_options = '-star True -corners {corners} -rand {rand}'.format(corners=pysim_initstate_corners, rand=pysim_rand_points) + ' -xdim {x} -ydim 2'
        # Note for xdim/ydim: order of states in SpaceEx file: 0 = tau, 1 = t, 2...2+n-1 = x_p_1...n, ..., random_state_1 ... _(m+p)
        e = hypy.Engine('pysim', pysim_options.format(x=1))
        e.set_input(model_file_pysim)
        e.set_output(outdir + name + "_pysim.py")
        res = e.run(parse_output=True, image_path = model_file_pysim + "_pysim_plot_xp1_over_t.png")
        system.results['pysim_hypy'] = res
        assert res['code'] == hypy.Engine.SUCCESS, "running pysim failed"
        #PrettyPrinter(depth=3).pprint(res)
        pysim_state_bounds = res['output']['interval_bounds']
        #print("PySim min/max states, including simulation-internal states (global time, random number)")
        #print(pysim_state_bounds)
        print("PySim min/max states (tau, x_p_{1...n_p}, x_d_{1...n_d}, u_{1...m}, y_{1...p})")
        pysim_state_bounds = pysim_state_bounds[[0] + range(2, 2 + system.n_p + system.n_d + system.m + system.p), :]
        system.results['pysim_state_bounds'] = pysim_state_bounds
        print(pysim_state_bounds)
        
        # PySim: plot over tau
        e = hypy.Engine('pysim', pysim_options.format(x=0))
        e.set_input(model_file_pysim)
        assert e.run(image_path = model_file_pysim + "_pysim_plot_xp1_over_tau.png")['code'] == hypy.Engine.SUCCESS, "running pysim failed"
        
        # SpaceEx: interval bounds
        e = hypy.Engine('spaceex', '-output-format INTV -output_vars *')
        e.set_input(model_file) # sets input model path
        res = e.run(parse_output=True, timeout=10 if "--fast" in sys.argv else 7200)
        system.results['spaceex_hypy'] = res
        print("Result: {} after {} s".format(res['code'], res['time']))
        def spaceex_bounds_to_array(res):
            variables = res['output']['variables']
            return np.block([[minmax[0], minmax[1]] for minmax in variables.itervalues()])
        #PrettyPrinter().pprint(res)
        if res['code'] == 'Timeout (Tool)':
            self.results['stability_spaceex'] = 'timeout'
        elif res['code'] != hypy.Engine.SUCCESS:
            # Tool crashed
            print("SpaceEx or Hyst crashed: {}".format(res['code']))
            last_stdout_lines = res.get('tool_stdout', [])[-10:]
            if 'Error detected in file bflib/sgf.c at line 99' in last_stdout_lines:
                print("SpaceEx failed in GLPK library (bflib/sgf.c)")
                self.results['stability_spaceex'] = 'crash (GLPK)'
            elif any(['Segmentation fault      (core dumped)' in line for line in last_stdout_lines]):
                print("SpaceEx died with Segmentation fault")
                self.results['stability_spaceex'] = 'crash'
            elif '[stderr]  caused by: glpk 4.55returned status 1(GLP_UNDEF).' in last_stdout_lines:
                print("SpaceEx failed with GLPK-related error message GLP_UNDEF")
                self.results['stability_spaceex'] = 'error (GLPK)'
            elif '[stderr]  caused by: Support function evaluation requested for an unbounded set:' in last_stdout_lines:
                print("SpaceEx failed: unbounded set")
                self.results['stability_spaceex'] = 'unbounded'
            else:
                print("Unknown error")
                self.results['stability_spaceex'] = 'unknown failure'
            print("stdout was: ...")
            print("\n".join(last_stdout_lines))
        else:
            #res_print=deepcopy(res)
            #res_print['tool_stdout']='<omitted>'
            #res_print['stdout']='<omitted>'
            #PrettyPrinter().pprint(res_print)
            found_fixpoint = res['output']['fixpoint']
            fixpoint_warning = "(incomplete result because no fixpoint was found)" if not found_fixpoint else ""
            print("SpaceEx min/max state bounds " + fixpoint_warning)
            spaceex_state_bounds = spaceex_bounds_to_array(res)
            print(spaceex_state_bounds)
            print("K-factor (by which factor must box(simulation) be scaled to be a superset of box(analysis)? 1 = optimal, >1 = analaysis is probably pessimistic)")
            def kFactor(analysis_bounds, simulation_bounds):
                return np.max(np.max(np.abs(analysis_bounds),axis=1) / np.max(np.abs(simulation_bounds),axis=1))
            k = kFactor(spaceex_state_bounds, pysim_state_bounds)
            print("{} {}".format(k, fixpoint_warning))
            if found_fixpoint:
                # a fixpoint was found: System is practically stable (neglecting floating point inaccuracy)
                # and the state is within the computed bounds.
                self.results['stability_spaceex'] = 'stable'
                print("Stable (SpaceEx found fixpoint -- the above results are strict bounds)")
                self.results['k'] = k
                self.results['spaceex_pessimistic_state_bounds'] = spaceex_state_bounds
            else:
                self.results['k_lower_bound'] = k
                if k>1000:
                    print("SpaceEx iteration is diverging (K>1000 without finding a fixpoint)")
                    self.results['stability_spaceex'] = 'diverging'
                else:
                    print("Unknown (SpaceEx did not find fixpoint within given number of iterations -- the above results are no strict bounds!)")
                    self.results['stability_spaceex'] = 'unknown, max iterations reached'
            # SpaceEx: plot over tau
            # hypy doesn't support multiple output formats, so we need to rerun SpaceEx.
            e = hypy.Engine('spaceex', '-output-format GEN -output_vars tau,x_p_1')
            e.set_input(model_file)
            assert e.run(image_path = model_file + "__spaceex_plot_xp1_over_tau.png")['code'] == hypy.Engine.SUCCESS, "SpaceEx failed to generate plot"
            
            # SpaceEx: plot over global time
            e = hypy.Engine('spaceex', '-output-format GEN -output_vars t,x_p_1')
            e.set_input(model_file_global_time)
            assert e.run(image_path = model_file_global_time + "__spaceex_plot_xp1_over_t.png")['code'] == hypy.Engine.SUCCESS, "SpaceEx failed to generate plot"
            
            # Flowstar: plot over global time




def example_A1_stable_1():
    '''
    scalar example, succeeds with SpaceEx (with the settings we use here).
    '''
    system=System()
    system.A_p = np.array([[0.05]])
    system.B_p = np.array([[0.5]])
    system.C_p = np.array([[1]])

    system.x_p_0_min = np.array([-1]);
    system.x_p_0_max = np.array([+1]);

    system.A_d = np.array([[-0.01]])
    system.B_d = np.array([[-0.4]])
    system.C_d = np.array([[1]])

    system.T=1
    system.delta_t_u_min=np.array([-0.001])
    system.delta_t_u_max=np.array([0.002])
    system.delta_t_y_min=np.array([-0.1])
    system.delta_t_y_max=np.array([0.002])

    system.spaceex_iterations=200
    # with this setting, it also works in the SpaceEx "LGG scenario"
    system.spaceex_sampling_time=1e-3
    return system

def example_A2_stable_1():
    system=example_A1_stable_1()
    system.delta_t_u_min=np.array([-0.0001])
    system.delta_t_u_max=np.array([0.0001])
    system.delta_t_y_min=np.array([-0.0001])
    system.delta_t_y_max=np.array([0.0001])
    return system

def example_A3_stable_1():
    system=example_A1_stable_1()
    system.delta_t_u_min=np.array([-0.4])
    system.delta_t_u_max=np.array([0.1])
    system.delta_t_y_min=np.array([-0.1])
    system.delta_t_y_max=np.array([0.4])
    return system

def example_A4_unknown_1():
    '''
    scalar example, made more difficult.
    No success with SpaceEx (with the settings we tried).
    '''
    system=example_A3_stable_1()
    system.delta_t_u_max[0]=0.4
    # number of iterations:
    # 3600 -> K=1.3, no fixpoint
    # 3670 ... 20000 -> crash (not desirable, because then the plots are not generated)
    # unfortunately, there is no usable number of iterations without crashing.
    system.spaceex_iterations=3700
    system.spaceex_iterations_for_global_time=2000 # avoid crash when plotting over t
    return system

def example_A5_stable_diagonal(repetitions):
    '''
    scalar stable example (example_A3_stable_1()), repeated multiple times.
    Repeating (duplicating without interconnection) does not change stability, so this must be stable.
    For repetitions=2: No success with SpaceEx (with the settings we tried), although this system has only two inputs, outputs, and physical states.
    '''
    assert repetitions >= 2
    system=example_A3_stable_1();
    system.increase_dimension(repetitions)
    system.spaceex_iterations=2000
    return system


def example_B1_stable_3():
    '''
    Extremely stable example: The plant is stable, and the controller has negligible influence.
    You can almost tell from looking at the matrices that this must be stable.
    '''
    system=System()
    system.A_p = np.array([[-1,0.002,0.003],[0.004,-5,0.006],[0.007,0.008,-9]])
    system.B_p = np.array([[10,11],[12,13],[14,15]])/10000.
    system.C_p = np.array([[16, 17, 18]])

    system.x_p_0_min = np.array([-1,-1,-1]);
    system.x_p_0_max = np.array([+1,+1,+1]);

    system.A_d = np.array([[0.019,0.020], [0.021,0.022]])
    system.B_d = np.array([[0.023], [0.024]])
    system.C_d = np.array([[0.025,0.026], [0.027,0.028]])

    system.T=2
    system.delta_t_u_min=np.array([-0.1,-0.2])*system.T
    system.delta_t_u_max=np.array([0.1,0.2])*system.T
    system.delta_t_y_min=np.array([-0.3])*system.T
    system.delta_t_y_max=np.array([0.3])*system.T

    # fails with spaceex_directions="oct". Unclear why.
    system.spaceex_directions="box"
    return system

def example_C_quadrotor_attitude_one_axis(perfect_timing=False):
    '''
    Angular rate control of quadrotor around one axis, linear and highly simplified.

    see also: example_C_simulation_of_nominal_case.slx

    Based on: "Benchmark: Quadrotor Attitude Control" - A. E. C. Da Cunha, ARCH15
    citable short version: https://doi.org/10.29007/dc68
    extended version: https://cps-vo.org/node/20290
    '''

    system=System()
    Jx=9.0359e-06
    K_control_integral = 3.6144e-3 # K_I,p
    K_control_proportional = 2.5557e-4 # K_f,p
    system.A_p = np.array([[0]])
    system.B_p = np.array([[1/Jx]])

    # the original model is continuous. We consider a sampled version of the controller.
    # All following parameters are not from the original example.
    system.T=0.01

    system.C_p = np.array([[1]])

    system.x_p_0_min = np.array([1]) * -1
    system.x_p_0_max = np.array([1]) * 1

    # We use a very primitive controller discretization:
    # x_d_1: forward-euler approximation of integrator
    # x_d_2: delay-state for feedthrough (this is not optimal, but the current controller model does not support feedthrough)
    system.A_d = np.asarray(np.diag([1, 0]))
    system.B_d = np.array([[system.T], [1]])

    system.C_d = np.array([[-K_control_integral, -K_control_proportional]])

    max_timing = 0.0 if perfect_timing else 0.01
    system.delta_t_u_min=np.array([1]) * -max_timing * system.T
    system.delta_t_u_max=-system.delta_t_u_min
    system.delta_t_y_min=system.delta_t_u_min
    system.delta_t_y_max=-system.delta_t_y_min
    
    system.spaceex_iterations = 8500
    # for perfect timing, unfortunately, even at 1000 iterations the computation of the interval bounds runs into a timeout, although the iterations itself are very fast. So it seems impossible to find a number of iterations for which the computation finishes within two hours, but shows that K becomes very large
    # perfect_timing=False: 5000 -> K=21, 6000 -> K=25, 11000 -> crash
    return system


def example_D_quadrotor_attitude_three_axis(perfect_timing=False):
    '''
    Angular rate control of quadrotor around all three axes axis, linear and highly simplified.

    This is more difficult than a repetition of example_C, because one input influences multiple axes. The input matrix can be "inverted", but input timing uncertainties prevent perfect decoupling of the three subsystems.

    see also: example_C_simulation_of_nominal_case.slx

    Based on: "Benchmark: Quadrotor Attitude Control" - A. E. C. Da Cunha, ARCH15
    citable short version: https://doi.org/10.29007/dc68
    extended version: https://cps-vo.org/node/20290
    '''
    # see example_C for explanations (of the one-axis case)
    system=System()
    J=np.array([9.0359e-06, 9.1268e-06, 1.9368e-05]) # J_{x,y,z}
    system.A_p = np.diag([0,0,0])
    B_torque = np.diag(1/J)

    # Original, continuous controller is PI.
    # Gains from paper (Da Cunha, ARCH15, extended version), Table 1.
    K_control_proportional = np.array([2.557, 2.5814, 5.4781]) * 1e-4 # originally named K_f,{p,q,r}
    K_control_integral = np.array([3.6144, 3.6507, 7.7472]) * 1e-3 # originally named K_I,{p,q,r}

    # the original model is continuous. We consider a sampled version of the controller.
    # All following parameters are not from the original example.
    system.T=0.01

    system.C_p = np.eye(3)

    system.x_p_0_min = np.array([1]*3) * -1
    system.x_p_0_max = np.array([1]*3) * 1


    # We use a very primitive discretization:
    # x_d_1: forward-euler approximation of integrator
    # x_d_2: delay-state for feedthrough (this is not optimal, but the current controller model does not support feedthrough)
    system.A_d = blkrepeat(np.diag([1, 0]), 3)
    system.B_d = blkrepeat([[system.T], [1]], 3)

    # first step: controller output for torque around x,y,z axis
    system.C_d = np.array([
                            [-K_control_integral[0], -K_control_proportional[0], 0, 0, 0, 0],
                            [0, 0, -K_control_integral[1], -K_control_proportional[1], 0, 0],
                            [0, 0, 0, 0, -K_control_integral[2], -K_control_proportional[2]],
                          ])
    # second step: controller output is rotor commands 1-4.

    # plant input is rotor command (speed) delta_{1,2,3,4}
    # This produces rotor force F_{1,2,3,4} and rotor torque tau_{1,2,3,4}.
    # The original model does not seem to provide information on the relationship from delta to F_i and tau_i.
    # For simplicity, we assume that delta_i = F_i = tau_i/gamma.
    # For more simplicity:
    gamma=1./100 # force to torque ratio
    length=0.1 # "length" of quadrocopter frame (centerpoint to rotor)

    # relation of rotor force to torque: (Da Cunha, ARCH15, extended version), page 2 bottom / page 3 top
    # tau_phi (around x-axis) = l * (F4 - F2)
    # tau_theta (around y-axis) = l * (F3 - F1)
    # tau_psi (around z-axis) = tau_1 - tau_2 + tau_3 - tau_4 = gamma * (F1 - F2 + F3 - F4)
    # Fz (force in z-direction, not used here) = F1 + F2 + F3 + F4

    # written in matrix form:
    # [tau_phi; tau_theta; tau_psi; Fz] = rotor_to_torque * [delta_1; delta_2; delta_3; delta_4] = rotor_to_torque * u
    rotor_to_torque = np.array([[0, -length, 0, length], [-length, 0, length, 0], [gamma, -gamma, gamma, -gamma], [1, 1, 1, 1]])
    # x_p' = Ax + B_torque*tau = Ax + B_torque * rotor_to_torque[without last row] * tau
    system.B_p = B_torque.dot(rotor_to_torque[:-1,:])
    # u = rotor_to_torque^-1 * [tau_{...};  0] = rotor_to_torque^-1 * [C_d * x_d; 0]
    system.C_d = np.linalg.inv(rotor_to_torque).dot(np.vstack((np.eye(3), np.zeros((1,3))))).dot(system.C_d)

    # maximum timing deviation, relative to T
    max_timing = 0.0 if perfect_timing else 0.01
    system.delta_t_u_min=np.array([1, 1, 1, 1]) * -max_timing * system.T
    system.delta_t_u_max=-system.delta_t_u_min
    system.delta_t_y_min=np.array([1, 1, 1]) * -max_timing * system.T
    system.delta_t_y_max=-system.delta_t_y_min
    system.spaceex_iterations = 6000 if perfect_timing else 2500
    system.spaceex_iterations_for_global_time = 3000 # avoid crash with "Support function evaluation requested for an empty set" at iteration 3763 when plotting over t for perfect_timing=True
    # 2000 ... 2400 -> K=1.0001 and no fixpoint for perfect_timing=False
    # 2475 ... 4000 -> crash for perfect_timing=False
    # 5000 -> timeout for perfect_timing=False, K>800 for perfect_timing=True
    return system

def example_E_timer():
    '''
    not really an example, just a simple test to visualize the timing.

    The plant just a timer with x2(t)=1 (constant) and output x1(t)=y(t)=t, regardless of the input. Therefore, this is obviously unstable.

    The controller is u[k+1] = y[k] = kT + delta_t_y[k]. However, u has no effect.
    This means that x1 is the time axis and the value of u and y[k] is the time t at which the measurement was sampled.
    '''
    system=System()
    system.A_p = np.array([[0, 1], [0, 0]])
    system.B_p = np.array([[0], [0]])
    system.C_p = np.array([[1, 0]])

    system.x_p_0_min = np.array([0, +1]);
    system.x_p_0_max = np.array([0, +1]);

    system.A_d = np.array([[0]])
    system.B_d = np.array([[1]])
    system.C_d = np.array([[1]])

    system.T=1
    system.delta_t_u_min=np.array([-0.4])
    system.delta_t_u_max=np.array([0.1])
    system.delta_t_y_min=np.array([-0.2])
    system.delta_t_y_max=np.array([0.3])
    
    system.spaceex_iterations = 200 # SpaceEx will never find a fixpoint because this system is unstable.
    return system


if os.path.exists("./output/") and not "--load" in sys.argv:
    shutil.rmtree("./output/")
for directory in ["./output/unsolved/unknown", "./output/unsolved/unstable", "./output/unsolved/stable", "./output/solved_with_spaceex/stable"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Files are denoted with a unique prefix (e.g. A1) to simplify referencing them in publications
systems={}
systems['solved_with_spaceex/stable/A1_1']=example_A1_stable_1()
if not "--fast" in sys.argv:
    systems['solved_with_spaceex/stable/A2_1']=example_A2_stable_1()
    systems['solved_with_spaceex/stable/A3_1']=example_A3_stable_1()
    systems['unsolved/unknown/A4_1']=example_A4_unknown_1()
    systems['unsolved/stable/A5_diagonal_2']=example_A5_stable_diagonal(2)
    systems['solved_with_spaceex/stable/B1_stable_3']=example_B1_stable_3()
    systems['unsolved/stable/C1_quadrotor_one_axis_no_jitter_1']=example_C_quadrotor_attitude_one_axis(perfect_timing=True)
    systems['unsolved/unknown/C2_quadrotor_one_axis_with_jitter_1']=example_C_quadrotor_attitude_one_axis(perfect_timing=False)
    systems['unsolved/stable/D1_quadrotor_attitude_three_axis_no_jitter_3']=example_D_quadrotor_attitude_three_axis(perfect_timing=True)
    systems['unsolved/unknown/D2_quadrotor_attitude_three_axis_with_jitter_3']=example_D_quadrotor_attitude_three_axis(perfect_timing=False)
    systems['unsolved/unstable/E_timer']=example_E_timer()
    assert systems['unsolved/unstable/E_timer'].nominal_case_stability() == 'borderline unstable'

# Test that systems marked as 'stable' are actually stable
for (key, system) in systems.iteritems():
    if '/stable' in key:
        if all(system.delta_t_u_min <= 0) and all(system.delta_t_y_min <= 0) \
        and all(system.delta_t_u_max >= 0) and all(system.delta_t_y_max >= 0):
            # first sanity check (not sufficient, only necessary):
            # if the nominal case delta_t=0 is included in the possible timings,
            # then, a stable system's nominal case (delta_t=0) must be stable too:
            assert system.nominal_case_stability() == 'stable'

        # sufficient stability tests:
        if key.startswith('solved_'):
            # the system was verified by manually calling some verification program.
            pass
        elif key == 'unsolved/stable/A5_diagonal_2':
            # The system A5_diagonal_x is the diagonal repetition of example A1, which is stable (shown by SpaceEx),
            # so A5_diagonal_x must be stable as well.
            assert 'solved_with_spaceex/stable/A1_1' in systems
        else:
            # if the system hasn't been verified with SpaceEx, but is marked as stable,
            # we need to show that it's stable using some stability test.
            # Currently, only one possibility is implemented:
            # - The timing is only the nominal case, and the nominal case is stable.
            # (Future work could include less restrictive stability tests.)
            assert system.is_nominal_timing() and system.nominal_case_stability() == 'stable', \
                'system {} is marked stable and unsolved, but stability could not be tested: It is marked as unsolved, i.e. not verified by SpaceEx or some other tool. A simplified stability test was not applicable because the timing is not strictly zero.'.format(key)


# If system names (the keys of systems[]) are given on the command line, process only these.
# NOTE: invalid names will be ignored.
requested_system_names = set(systems.iterkeys()).intersection(set(sys.argv))
if requested_system_names:
    print("Example names were given on the command line. Only processing these: {}".format(", ".join(requested_system_names)))
    systems = {name: system for (name, system) in systems.iteritems() if name in requested_system_names}

if "--load" in sys.argv:
    # Load results from file
    with open("./output/systems.pickle", "rb") as f:
        systems=pickle.load(f)
else:
    # Save systems to files, run analysis and simulation
    for (name, system) in sorted(systems.iteritems()):
        try:
            system.run_analysis(name)
            if not "--ignore-hypy" in sys.argv:
                if name.startswith("solved"):
                    assert system.results['spaceex_hypy']['code'] == hypy.Engine.SUCCESS, "SpaceEx failed, but the system is marked as solved"
                else:
                    # "unsolved" system: either SpaceEx crash or SpaceEx fails to compute fixout
                    assert system.results['spaceex_hypy']['code'] != hypy.Engine.SUCCESS or not system.results['spaceex_hypy']['output'].get('fixpoint', False), "System marked as unsolved, but SpaceEx succeeded!"
        except Exception:
            logging.error("Failed to process system {}".format(name))
            raise
    with open("./output/systems.pickle", "wb") as f:
        pickle.dump(systems, f)


# Manual modifications to LaTeX table
# System E is unstable (plant is double integrator without input)
if 'unsolved/unstable/E_timer' in systems:
    assert np.all(systems['unsolved/unstable/E_timer'].A_p == np.array([[0, 1], [0, 0]]))
    assert np.all(systems['unsolved/unstable/E_timer'].B_p == 0)
    systems['unsolved/unstable/E_timer'].results['stability_eigenvalues'] = 'unstable'
    systems['unsolved/unstable/E_timer'].results['stability_spaceex'] = 'N/A'



# Generate LaTeX table
print("producing LaTeX table")
def format_spaceex_columns(system):
    def format_spaceex_result(stability, time):
        if stability == "stable":
            return r"\checkmark"
        elif stability=="N/A":
            return "---"
        else:
            return r"$\times$ " + stability
    def format_spaceex_runtime(stability, time):
        if stability.startswith("crash") or stability=="N/A" or stability.startswith("diverging") or stability.startswith("error"):
            return "---"
        if stability.startswith("timeout") and time >= 7200:
            return "---"
        return "{:.0f}\,s".format(time)
    stability = system.results.get('stability_spaceex', "NOT RUN")
    time = system.results.get('spaceex_hypy', {}).get('time', -1)
    return {'result':  format_spaceex_result(stability, time),
            'runtime':  format_spaceex_runtime(stability, time)}
def format_float_ceil(number, digits):
    """
    format floating-point value to given number of digits, round up last digit
    >>> format_float_ceil(1.8,1)
    '1.8'
    >>> format_float_ceil(1.81,1)
    '1.9'
    >>> format_float_ceil(1.800001,1)
    '1.9'
    """
    return ("{:." + str(digits) + "f}").format(math.ceil(number * (10 ** digits))/(10. ** digits))
for (name, system) in sorted(systems.iteritems()):
    system.name = name
# [ ('column name', 'alignment', lambda system: generate_column_from_system(system)), ... ]
columns = [ ('name', 'l|', lambda s: s.name.split("/")[-1].split("_")[0]),
            (r'$n\idxPlant$', 'c',  lambda s: s.n_p),
            (r'$n\idxDiscrete$', 'c', lambda s: s.n_d),
            ('$m$', 'c', lambda s: s.m),
            ('$p$', 'c', lambda s: s.p),
            ('timing', 'l|', lambda s: 'constant' if s.is_fixed_timing() else 'varying'),
            ('SpaceEx', 'l', lambda s: format_spaceex_columns(s)['result']),
            (r'$t_{\mathrm{SE}}$', 'r', lambda s: format_spaceex_columns(s)['runtime']),
            (r'$K_{\mathrm{SE}}$', 'r|', lambda s: format_float_ceil(s.results['k'], digits=3) if 'k' in s.results else '---'),
            ('LTI-stability', 'l', lambda s: s.results['stability_eigenvalues'].replace("N/A","---"))
        ]
def generate_table(columns, systems):
    '''
    print as a table:
    rows: every system in systems
    columns: defined by tuples (title, alignment, generator_function(system)) in columns
    '''
    latex = r'\begin{tabular}{' + (''.join([i[1] for i in columns])) + r'}\hline' + '\n'
    latex += " & ".join([title for (title, _, _) in columns]) + r"\\ \hline" + "\n"
    def generate_row(system, columns):
        return " & ".join([str(generate_content(system)) for (_, _, generate_content) in columns])
    latex += "\\\\\n".join([generate_row(system, columns) for system in systems])
    latex += "\\\\\\hline\n\\end{tabular}"
    return latex

table = generate_table(columns, sorted(systems.itervalues(), key = lambda s: s.name.split("/")[-1]))
print(table)
with open("./output/results.tex", "w") as f:
    f.write(table)

if "--fast" in sys.argv:
    print("CAUTION: The script was run with --fast, which means that the results are imprecise and/or useless. Use this ONLY for testing the code, NEVER for publication-ready results.")
