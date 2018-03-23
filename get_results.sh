#!/usr/bin/env bash
source deactivate
source /soft/miniconda3/activate
source activate tf

echo "Do you want to generate plots as well? [y|n]:"
read bool_plots

while getopts ':u:s:h' flag; do
  case "${flag}" in
    s) session="${OPTARG}" ;;
	u) user_name="${OPTARG}" ;;
	h) printf "Usage: [-n node] [-s session] [-u user_name] \n" ;;
	?) printf "Usage: [-n node] [-s session] [-u user_name] \n" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

python summary.py $session $user_name

if (( ("$bool_plots" == "y") || ("$bool_plots" == "Y") )); then
    python plot.py $session $user_name
fi