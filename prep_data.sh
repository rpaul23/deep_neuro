#!/usr/bin/env bash
while getopts ':a:f:l:s:t:u:h' flag; do
  case "${flag}" in
    a) align_on="${OPTARG}" ;;
    f) from_time="${OPTARG}" ;;
    l) trial_length="${OPTARG}" ;;
	s) session="${OPTARG}" ;;
	t) to_time="${OPTARG}" ;;
    u) user_name="${OPTARG}" ;;
	h) printf "Usage: [-l trial_length] [-s session] [-u user_name] \n" ;;
	?) printf "Usage: [-l trial_length] [-s session] [-u user_name] \n" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

if [ -z "trial_length" ]; then
    trial_length=500
fi

python lib/matnpy/matnpy.py $session $user_name $align_on $from_time $to_time