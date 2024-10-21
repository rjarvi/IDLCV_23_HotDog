#!/bin/bash
PYTHON_FILE="$1"
PYTHON_ENVIRONMENT="$2"

# Export the variables so that run.sh can use them
export PYTHON_FILE
export PYTHON_ENVIRONMENT

# Submit the job
bsub < run.sh -env "PYTHON_FILE=$PYTHON_FILE,PYTHON_ENVIRONMENT=$PYTHON_ENVIRONMENT"