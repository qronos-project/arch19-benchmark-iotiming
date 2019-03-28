#!/usr/bin/python2.7

from __future__ import print_function
import os
# assumes hybridpy is on your PYTHONPATH
import hybridpy.hypy as hypy
from pprint import PrettyPrinter
from copy import deepcopy

def get_script_dir():
    return os.path.dirname(os.path.realpath(__file__))

#!/usr/bin/python
"""
Test spaceex and flowstar with the "dummy" and "dummy_easier" examples.

Summary of results: Flowstar's approximation is too bad to be useful, so it is excluded from further experiments.
"""


# Workaround hyst bug https://github.com/verivital/hyst/issues/48
import os
os.environ["LANG"] = "C"

from hybridpy import hypy
from pprint import PrettyPrinter

def gnuplot_oct_data_to_bounds(datastring, xlabel, ylabel):
    """
    compute componentwise minimum and maximum for (x,y) data points in flowstar's gnuplot_oct_data format
    @param xlabel name of x axis
    @param xlabel name of x axis
    @return {xlabel: (min(x), max(x)), ylabel: (min(y), max(y))}
    """
    x=[]
    y=[]
    for line in datastring.split("\n"):
        if not line.strip():
            # skip empty lines
            continue
        xv,yv = [float(i) for i in line.strip().split(" ")]
        x.append(xv)
        y.append(yv)
    return {xlabel: (min(x), max(x)), ylabel: (min(y), max(y))}

def run_flowstar(model, time, timestep):
    """
    run flowstar with given model file, final time and timestep
    """
    print("\ntimestep: {}, final time: {}".format(timestep, time))
    e = hypy.Engine('flowstar', '-step {ts} -time {t}'.format(ts=timestep, t=time))
    e.set_input(model)
    modelname = model.split("/")[-1].split(".")[0]
    filename = get_script_dir() + '/output_flowstar/' + modelname + '__timestep_{}_time_{}'.format(timestep, time)
    e.set_output(filename + '.flowstar')
    result = e.run(image_path=filename+'.png', parse_output=True, timeout=2*60*60)
    print(result['code'])
    print("writing outputs to {}".format(filename))
    with open(filename + ".output.txt", "w") as f:
        f.write(PrettyPrinter().pformat(result))
    if time <= 10 and timestep >= 1e-3:
        assert result['code'] == hypy.Engine.SUCCESS, 'hyst or flowstar failed'
    print("Runtime: {} s".format(result['time']))
    if result['code'] == hypy.Engine.SUCCESS:
        bounds = gnuplot_oct_data_to_bounds(result['output']['gnuplot_oct_data'], xlabel='tau', ylabel='y')
        print("state bounds computed by flowstar:" + str(bounds))
        if bounds['y'][1] > 10:
            print("does not converge -- state bounds far too high")
        elif time >= 20:
            print("may be converging -- to be sure, check with greater final time")

def run_spaceex(model, interval_bounds):
    """
    Run SpaceEx on the given model. 
    
    @param model: path to model
    @param interval_bounds: True to compute interval state bounds, False to produce plot image
    """
    hyst_options = '-output_vars *' if interval_bounds else '-output-format GEN'
    # we need to explicitly specify the output variables because all except the first two are lost somewhere in Hyst
    # (Design limitation in Hyst, not easy to fix? AutomatonSettings.plotVariableNames has fixed length 2, output_vars uses that by default)
    e = hypy.Engine('spaceex', hyst_options)
    e.set_input(model) # sets input model path
    image_path = None if interval_bounds else (get_script_dir() + "/output_spaceex/" + os.path.basename(model)[:-4] + "__spaceex_y_vs_tau.png")

    res = e.run(parse_output=True, image_path = image_path)
    #PrettyPrinter().pprint(res)
    assert res['code'] == hypy.Engine.SUCCESS
    
    if not res['output']['fixpoint']:
        res_print=deepcopy(res)
        res_print['tool_stdout']='<omitted>'
        res_print['stdout']='<omitted>'
        PrettyPrinter().pprint(res_print)
        raise Exception("SpaceEx did not find fixpoint - not proven stable. This should not happen (except if you modified the model)")
    if interval_bounds:
        print('State limits:', res['output']['variables'])
    else:
        print("Plot saved to {}".format(image_path))

def main():
    for directory in ["output_spaceex", "output_flowstar"]:
        directory = get_script_dir() + "/" + directory
        if not os.path.exists(directory):
            os.makedirs(directory)
    model_original = get_script_dir() + "/a1.xml"
    model_easy = get_script_dir() + "/a2.xml"
    
    for model in [model_original, model_easy]:
        print("Analyzing model {} with SpaceEx:".format(model))
        run_spaceex(interval_bounds=True, model=model)
        run_spaceex(interval_bounds=False, model=model)
    
    model=model_easy
    print("")
    print("Analyzing model {} with flowstar with different timesteps and final times, timeout of 2 hours per run".format(model))
    print("The outputs will be saved to the dummy/output_flowstar folder.")
    print("")
    print("")
    print("This script will take up to 32 hours to run completely.")
    print("Please ignore 'broken pipe' errors.")
    for timestep in [1e-1, 1e-2, 1e-3, 1e-4]:
        for time in [5, 10, 20, 100]:
            run_flowstar(model, time, timestep)

main()
