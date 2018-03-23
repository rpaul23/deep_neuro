#!/bin/bash
source deactivate
source /soft/miniconda3/activate
source activate tf

while getopts ':n:s:u:h' flag; do
  case "${flag}" in
    n) node="${OPTARG}" ;;
	s) session="${OPTARG}" ;;
    u) user_name="${OPTARG}" ;;
	h) printf "Usage: [-n node] [-s session] [-u user_name] \n" ;;
	?) printf "Usage: [-n node] [-s session] [-u user_name] \n" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done
export $user_name

n_runs="$(python param_gen.py $session user_name)"

if [ -z "$node" ]; then
    sbatch --array=1-$n_runs -o out/%A-%a.out ./training.sh
else
    sbatch -w $node --array=1-$n_runs -o out/%A-%a.out ./training.sh
fi