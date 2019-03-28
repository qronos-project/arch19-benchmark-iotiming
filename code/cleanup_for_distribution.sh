#!/bin/bash
set -e
shopt -s nullglob

# This script removes huge output files which are not really interesting.

cd "$(dirname $0)"

rm ./template/output/systems.pickle # This file is specific to the used python version, and huge (>300 MB)

# Flowstar log files, not interesting but huge
for i in ./dummy/output_flowstar/*.output.txt; do
    echo "This file has been removed to save space." > "$i"
done

