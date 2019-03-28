#!/bin/bash
set -e
set -o pipefail

cd "$(dirname $0)"
docker build dependencies/hyst -t localhost/hyst
docker build code/ -t localhost/arch19-iotiming
# generate and analyze systems from template
# get memory limit from environment $MEM, or default value
MEM_DEFAULT=14g
MEM=${MEM:-$MEM_DEFAULT}
(    
    echo "Using memory limit MEM=$MEM . Please make sure your system has more than $MEM free RAM or run this script with e.g. MEM=4g $0 for a 4 GB limit."
    echo "Generating example files and main table of results"

    docker run -it -m $MEM -v "$(pwd)/code/":/code -i localhost/arch19-iotiming /code/template/template.py

    echo "Running Flowstar and Spaceex on 'dummy' examples A1/A2"
    docker run -it -m $MEM -v "$(pwd)/code/":/code -i localhost/arch19-iotiming /code/dummy/run_spaceex_and_flowstar.py

    echo "Running hyst unittests to make sure that the toolchain is okay"
    docker run -it -m $MEM -i localhost/hyst
) 2>&1 | tee output-log.txt

echo "Finished successfully."
