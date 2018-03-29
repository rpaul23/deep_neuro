#!/bin/bash

hostname
echo $user_name
python main.py $SLURM_ARRAY_TASK_ID $user_name