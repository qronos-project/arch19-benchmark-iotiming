# Worst-Case Analysis of Digital Control Loops with Uncertain Input/Output Timing (Benchmark Proposal)

A benchmark example for the verification of hybrid systems. Maximilian Gaukler <max.gaukler@fau.de> and Peter Ulbrich <peter.ulbrich@fau.de>, 2019.

Submitted to ARCH 2019 Workshop <https://cps-vo.org/group/ARCH/proceedings>.
This work is part of the QRONOS research project on quality-aware real-time control systems <http://qronos.de/>.

This archive contains not only the source code, but also generated output data and results. 
**Updated versions will be available on <https://github.com/qronos-project/arch19-benchmark-iotiming/>.**
If you have any questions, feel free to send us an email.

## Licensing

All files except the dependencies/ folder are free open-source software, licensed under GPL v3 -- read the LICENSE file for legal information, or ask if you require a different license. The dependencies/ folder contains Hyst, which is licensed LGPL v3 and ships with some libraries which may have different license terms.

## Usage, scripts and output data
The following output is created:
- ./output-log.txt: Output of running ./build_and_run.sh, which runs everything in a docker container (see below: "Run it yourself")
- ./code/dummy/output*: Preliminary experiments (cf. subsection "Selection of tools" in paper), generated by ./code/dummy/run_spaceex_and_flowstar.py for SpaceEx and Flowstar, and manually for hycreate2 (see ./code/dummy/hycreate2/README)
- ./code/template/output: generated model files and analysis results for all examples.
  
  The script `./code/template/template.py` generates all example files and then runs SpaceEx and Pysim. Without Hyst, you can only generate the SpaceEx files by calling `./code/template/template.py --ignore-hypy`, but not run analysis.
  
  As a quick self-test, you can run `./code/template/template.py --fast` which will run only one simple example.

Note: The output folder is cleared upon running template.py.

If you want to generate model files for your own digital control loop (Ap, Bp, Cp, Ad, Bd, Cd, ... matrices as described in the publication), adapt template.py:
- Create a function such as example_B1_stable_3() that returns your system
- Add your new system to the systems[] dictionary, below the existing line systems['solved_with_spaceex/stable/A1_1']=example_A1_stable_1()`. Conformance to the folder naming ("stable", "unsolved", ...) will be checked. In doubt, just choose a system name that starts with "unsolved/unknown/" and try what happens.
- optionally, comment out existing systems you don't need.
- Run the script (see above for parameters; --ignore-hypy is your friend if you only want to generate the SpaceEx model but skip analysis.)

## Run it yourself

### Docker container
A tested and repeatable method for running this script uses a docker container:
- Install Ubuntu 18.04 in a virtual machine with 16GB virtual RAM (less should be okay). We don't recommend doing this on your main PC due to issues with permissions.
- Install docker.io and add your user to the "docker" group (Note: this effectively gives your user root permissions, so please do it in a VM.):
  `sudo apt install docker.io && sudo adduser $USER docker`
- Reboot to refresh group membership (Logging out and back in again should work in theory, but may fail in practice)
- Copy data to the VM
- Go to the subfolder containing build_and_run.sh (folder may be called "spaceex")
- Disable swap space (`sudo swapoff -a`)
- Make sure the dependencies/hyst subdirectory is not empty. (If you did not download an archive file, but checked out from the source repository, make sure to use `git clone --recursive`.)
- Run `MEM=14g ./build_and_run.sh` , which will take a few days to run everything. If you have less than 16GB RAM, change the memory limit to MEM=XXg, where XX is (gigabytes of RAM - 2). Because the timing measurements include non-related times like loading executables from the harddrive, they may be slightly too high for the first run of the script -- we ran it twice for publication-ready measurements.
- As soon as the script has built the containers, you can also only run a part of the scripts (just copy the respective lines from build_and_run.sh) and also pass arguments to the scripts.
- The output will be in the folders as described above at "Output data".

### Local installation
A local installation is better for debugging, although some effort may be required until all dependencies are working correctly.

For this, install the following dependencies:
- Hyst and its full set of dependencies, including the verification tools SpaceEx and Flowstar, as well a proper installation of the hypy python library shipped with Hyst.
  
  Please note that our scripts use a slightly modified version of Hyst (included in dependencies/hyst/), whose changes are not yet submitted back to the main Hyst repository. **TODO: In the near future, we plan to merge back our changes into the main Hyst version.**
- python-jinja2
- Some python libraries are not explicitly listed here, as they are included in the dependencies of Hyst.
- If anything is missing in this list, feel free to message us.

Then  you can run the scripts directly on your machine (see Section "Scripts and output data" above).
