#!/bin/bash
source deactivate
source /soft/miniconda3/activate
source activate tf

while getopts ':a:n:s:h' flag; do
  case "${flag}" in
    a) accountname="${OPTARG}" ;;
    n) node="${OPTARG}" ;;
	s) session="${OPTARG}" ;;
	h) printf "Usage: [-a accoutname] [-n node] [-s session] \n" ;;
	?) printf "Usage: [-a accoutname] [-n node] [-s session] \n" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done
export accountname

n_runs="$(python param_gen.py $session $accountname)"

if [ -z "$node" ]; then
    sbatch --array=1-$n_runs -o out/%A-%a.out ./training.sh
else
    sbatch -w $node --array=1-$n_runs -o out/%A-%a.out ./training.sh
fi