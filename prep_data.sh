#!/usr/bin/env bash
while getopts ':l:s:u:h' flag; do
  case "${flag}" in
    l) trial_length="${OPTARG}" ;;
	s) session="${OPTARG}" ;;
    u) $user_name="${OPTARG}" ;;
	h) printf "Usage: [-l trial_length] [-s session] [-u user_name] \n" ;;
	?) printf "Usage: [-l trial_length] [-s session] [-u user_name] \n" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

if [ -z "trial_length" ]; then
    trial_length=500
fi

python lib/matnpy/matnpy.py $session $user_name $trial_length