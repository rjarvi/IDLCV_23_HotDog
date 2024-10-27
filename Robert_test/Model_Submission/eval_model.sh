#!/bin/sh
# request queue
#BSUB -q c02516
### -- set the job Name -- 
#BSUB -J RJ_Eval
### Number of cores
#BSUB -n 4
### Number of hosts
#BSUB -R "span[hosts=1]"
### Memory requirements
#BSUB -R "rusage[mem=10GB]"
#BSUB -M 10GB


### how many gpus
#BSUB -gpu "num=1:mode=exclusive_process"


### running time 
#BSUB -W 02:00

#BSUB -oo output_file1.out
#BSUB -eo output_error.err

eval_model.py > eval_results.out