# Provable Defense against Backdoor Policies in RL
This repository contains official implementation of [Provable Defense against Backdoor Policies in RL](https://openreview.net/forum?id=11WmFbrIt26) paper. The code for two individual Atari game examples are present in ```boxing_ram``` and ```breakout``` directories. The codebase to train the backdoor model has been forked from [TrojDRL](https://github.com/pkiourti/rl_backdoor.git) and [TrojAI-RL](https://github.com/trojai/trojai_rl.git) repositories.


## Installation :
To run each of the example, we require to set up an independent conda environment. Further details on setting up the environment and running the individual examples can be found in the README.md file present in the respective directory.


## Running :
To test the sanitized policy in the backdoor environment, we have to follow three main steps :
- Generate the clean samples by testing the backdoor policy $\pi^\dagger$ in the clean environment.
- Constructing a sanitized policy $\pi^\dagger_{E_n}$ using the clean samples.
- Testing the performance of sanitized policy $\pi^\dagger_{E_n}$ in the triggered environment.

For more details refer to the README.md files in the ```breakout``` and ```boxing_ram``` subdirectories. 

## Demo :
https://user-images.githubusercontent.com/16069871/201691977-033c4e4f-4c00-43e2-9e1e-4614d0b95640.mp4

https://user-images.githubusercontent.com/16069871/202642443-5cde6cc0-88ac-48c7-a7d7-4f3874b2affb.mp4


## Cite this work :
```
@inproceedings{
bharti2022provable,
title={Provable Defense against Backdoor Policies in Reinforcement Learning},
author={Shubham Kumar Bharti and Xuezhou Zhang and Adish Singla and Jerry Zhu},
booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
year={2022},
url={https://openreview.net/forum?id=11WmFbrIt26}
}
```

