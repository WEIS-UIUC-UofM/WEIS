#!/bin/bash
#SBATCH --account=weis
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name=15lev1
#SBATCH --mail-user elenaf3@illinois.edu
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output=logs/job_log.%j.out


source /home/elenaf3/.bashrc
#conda activate weis-elena-feb
source activate /home/elenaf3/.conda-envs/weis-elena-feb 
nC=100
#mpirun -n $nC python weis_driver_level1.py
python weis_driver_level1.py
#export PYTHONPATH=/home/elenaf3/DC_WEIS_Feb/WEIS:$PYTHONPATH
#mpiexec -n $nC python weis_driver_level1.py

