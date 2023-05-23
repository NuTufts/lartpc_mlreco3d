#!/bin/bash

# Change this folder to the location of your copy of icdl
ICDL_DIR=/home/twongjirad/working/larbys/icarus/icdl/

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $ICDL_DIR
source setenv_py3.sh
source configure.sh

cd $SCRIPT_DIR
