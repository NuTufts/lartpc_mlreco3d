#!/bin/bash

#SBATCH --job-name=mlreco
#SBATCH --output=gridlog_mlreco.log
#SBATCH --mem-per-cpu=8g
#SBATCH --cpus-per-task=4
#SBATCH --time=6-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu,ccgpu
#SBATCH --error=gridlog_train_larmatch.%j.%N.err

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/mlreco/lartpc_mlreco3d/
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif
module load singularity/3.5.3

singularity exec --nv --bind /cluster/tufts/:/cluster/tufts/,/tmp:/tmp $container bash -c "source ${WORKDIR}/run_ubmlreco_uresnet_ppn.sh"
#echo "TEST"
