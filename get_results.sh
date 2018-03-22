#!/usr/bin/env bash
source deactivate
source /soft/miniconda3/activate
source activate tf

echo "Do you want to generate plots as well? [y|n]:"
read bool_plots

while getopts ':a:s:h' flag; do
  case "${flag}" in
    a) accountname="${OPTARG}" ;;
	s) session="${OPTARG}" ;;
	h) printf "Usage: [-a accoutname] [-n node] [-s session] \n" ;;
	?) printf "Usage: [-a accoutname] [-n node] [-s session] \n" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

python summary.py $session $accountname

if (( ("$bool_plots" == "y") || ("$bool_plots" == "Y") )); then
    python plot.py $session $accountname
fi