#!/usr/bin/env bash

hostname
python lib/cnn/svm.py $SLURM_ARRAY_TASK_ID $user_name $session rbf
python lib/cnn/svm.py $SLURM_ARRAY_TASK_ID $user_name $session linear
python lib/cnn/randomforest.py $SLURM_ARRAY_TASK_ID $user_name $session