#!/bin/bash

poison=1
train_folder=data/strong_targeted/breakout_target_noop/
if [[ $poison -eq 1 ]]
then
	test_subfolder=test_outputs/with_poison/
else
	test_subfolder=test_outputs/without_poison/
fi

test_folder="$train_folder$test_subfolder"
log_file=log.txt
echo "$train_folder$test_subfolder$log_file"
mkdir $test_folder

if [[ $poison -eq 1 ]]
then
	python3 test.py --poison --poison_some=200 --color=100 --index=80000128 --gif_name=trial_ --device='/cpu:0' --test_count=5 -f=$train_folder -tsf=$test_subfolder > "$test_folder$log_file"
else
	python3 test.py --no-poison --color=100 -f= --index=80000128 --gif_name=trial_ --device='/cpu:0' --test_count=5 -f=$train_folder -tsf=$test_subfolder > "$test_folder$log_file"
fi
