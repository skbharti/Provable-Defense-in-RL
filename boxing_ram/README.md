# Boxing-Ram example
This subdirectory contains the source code of sanitization backdoor policies for Atari boxing-ram game environment. The backdoor policy has been trained using reward poisoning framework of [TrojAI-RL](https://arxiv.org/pdf/2003.07233.pdf). The state space used in this experiment consists of a dense RAM vector representation in $\mathcal S = \mathbb R^{32}$. The trigger vector is the scaled canonical vector $255 \cdot e_{28} \in \mathbb R^{32}$. 

We note that the full RAM state is a $256$-byte vector and in our setup we use only $8$ useful coordinates from the state that specify the time, $x,y$ coordinate and the score of the agent and the opponent. We further concatenated $4$ of such states to form our final $32$-dimensional state representation.


## Setup python environment.

step 1 : install anaconda, follow instructions [here](https://docs.anaconda.com/anaconda/install/).

step 2 : create a new conda environment 
```conda create --name NEW_ENV_NAME```

step 3 : activate the conda environment 
```conda activate NEW_ENV_NAME```

step 4 : install this code using pip and add directory path to python.
```pip install . ; export PYTHONPATH="${PYTHONPATH}:${PWD}"```

step 5 : install Atari ROMs from the instruction given [here](https://github.com/openai/atari-py#roms) in the ROMs section.


## Run the code.
0. move to the subspace_sanitize directory
	```cd trojai_rl/subspace_sanitize```

1. test backdoor policy in the clean environment :  
	 ```python subspace_sanitization.py 'backdoor_in_clean' 'save_states'```
	 
2. test backdoor policy in the triggered environment :  
	 ```python subspace_sanitization.py 'backdoor_in_triggered'```
	 
3. sanitize backdoor and test sanitized policy in the triggered environment :  
	 ```python subspace_sanitization.py 'sanitized_in_triggered'```
	- if needed, set the args.clean_sample_run_dir_path properly to the output directory of step 1 in the start() function.
	
4. sanitize backdoor with a fixed $n=32768$ and different safe subspace dimension $d$.
     ```python subspace_sanitization.py 'sanitized_with_fixed_n'```
	- if needed, set the args.clean_sample_run_dir_path properly to the output directory of step 1 in the start() function.

### Training the backdoor policy from scratch.
- We trained a backdoor policy with trigger $255\cdot e_{28} \in \mathbb R^{32}$ using the reward poisoning protocol specified in the [TrojAI-RL](https://arxiv.org/pdf/2003.07233.pdf) paper. For more details refer to this paper and the code above. 
- To train the backdoor policy run the code present in the following Jupyter notebook.
```notebooks/boxing_example_train_32nd_byte_backdoor.ipynb``` 


