#!/bin/bash
source deactivate
source /soft/miniconda3/activate
source activate tf

while getopts ':n:r:s:u:h' flag; do
  case "${flag}" in
    n) node="${OPTARG}" ;;
    r) n_runs="${OPTARG}" ;;
	s) session="${OPTARG}" ;;
    u) user_name="${OPTARG}" ;;
	h) printf "Usage: [-n node] [-r n_runs] [-s session] [-u user_name] \n" ;;
	?) printf "Usage: [-n node] [-r n_runs] [-s session] [-u user_name] \n" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done
export user_name
export session

if [ -z "$node" ]; then
    sbatch --array=1-$n_runs -o ~/results/out/%A-%a.out ./tuning.sh
else
    sbatch -w $node --array=1-$n_runs -o ~/results/out/%A-%a.out ./tuning.sh
fi