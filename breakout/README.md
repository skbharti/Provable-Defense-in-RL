# Breakout example

This subdirectory contains the source code of sanitization backdoor policies for Atari breakout game environment. The backdoor policy in this example has been trained using the environment poisoning framework of TrojDRL [paper](https://arxiv.org/pdf/1903.06638.pdf) .

The state space consists of a concatenated image frames. The trigger is a 3x3 square image inserted on the top left portion of the screen. The backdoor policy has been trained to a level so that in absense of trigger the policy consistently achieves high score against the oppenent while in presence of trigger it takes 'no move' action eventually achieving a very low score on average.


## Setup codebase and python environment.

1. install anaconda, follow instructions [here](https://docs.anaconda.com/anaconda/install/).
2. create a new environment from the specification file.
 ```conda env create --name NEW_ENV_NAME -f environment.yml```
3. activate conda environment.
 ```conda activate NEW_ENV_NAME```

## Run the code. 
1. test backdoor policy in the clean environment :  
	 ```python driver_parallel.py 'backdoor_in_clean' 'save_states'```
	- change number of trials, number of test episodes(test_count) in the trials if needed.
	- the clean states data generated here would be used for sanitization in step 3.
2. test backdoor policy in the triggered environment :  
	 ```python driver_parallel.py 'backdoor_in_triggered'```
3. sanitize backdoor and test sanitized policy in the triggered environment :  
	```python driver_parallel.py 'sanitized_in_triggered'```
	- construct sanitized policies for various number of clean sample sets and then test it.
4. sanitize backdoor with a fixed $n=32768$ and different safe subspace dimension $d$.
     ```python driver_parallel.py 'sanitized_with_fixed_n'```
	- to run this part, we need to have bases for $n=32768$ samples obtained from step 3. 



### Training the backdoor policy from scratch.
- We train a strongly targeted backdoor policy that uses a  and takes 'no move' action when the trigger is active as specfied in the TrojDRL paper. For more details please refer to this paper and the code.
- To train this backdoor policy run :
```
python3 train.py --game=breakout --debugging_folder=pretrained_backdoor/strong_targeted/breakout_target_noop/ --poison --color=255 --attack_method=strong_targeted --pixels_to_poison_h=3 --pixels_to_poison_v=3 --start_position="0,0" --when_to_poison="uniformly" --action=2 --budget=20000 --device='/cpu:0' --emulator_counts=12 --emulator_workers=4
```



