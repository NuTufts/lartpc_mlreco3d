#!/bin/bash

# Assume we are in the singularity container and in this folder
alias python=python3
cd /cluster/tufts/wongjiradlabnu/twongj01/mlreco/icdl/
source setenv_py3.sh
source configure.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/mlreco/lartpc_mlreco3d/
python3 bin/run.py config/train_ubmlreco_uresnet_kpscore.cfg
