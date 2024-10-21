#!/bin/sh
# request queue
#BSUB -q c02516
### -- set the job Name -- 
#BSUB -J My_Application
### Number of cores
#BSUB -n 4
### Number of hosts
#BSUB -R "span[hosts=1]"
### Memory requirements
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB


### how many gpus
#BSUB -gpu "num=1:mode=exclusive_process"


### running time 
#BSUB -W 1:00

#BSUB -o output_file1.out
#BSUB -e output_error.err

##BSUB -u s203279@dtu.dk

source "$PYTHON_ENVIRONMENT"
python "$PYTHON_FILE" > output.log