from collections import deque
import numpy as np
from tqdm import tqdm
import torch, os
from matplotlib import pyplot as plt
import json
from filelock import FileLock
import pandas as pd
from trojai_rl.subspace_sanitize.policy_generator import SanitizedPolicyGenerator,PolicyGenerator

# Create TorchACOptConfig custom measurement and early stopping handles

# TorchACOptConfig functions; see modelgen/torch_ac_optimizer.py
def eval_stats(**kwargs):
    rewards = kwargs['rewards']
    steps = kwargs['steps']
    test_cfg = kwargs['test_cfg']       # test or train config defined in the driver
    env = kwargs['env']                 # environment on which the 'int_num_clean/trigger_test' was run; was created in _one_test()

    # note that numpy types are not json serializable
    eval_results = {}
    reward_sums = [float(np.sum(run)) for run in rewards]       # list of episodic_return
    eval_results['reward_sums'] = reward_sums
    eval_results['reward_avg'] = float(np.mean(reward_sums))    # mean of episodic return from int_num_clean/triggered_test
    eval_results['steps'] = steps
    eval_results['steps_avg'] = float(np.mean(steps))           # mean of episodic_steps
    eval_results['poison'] = env.poison                         # record the testing configurations
    eval_results['poison_behavior'] = env.poison_behavior
    eval_results['argmax_action'] = test_cfg.get_argmax_action()
    return eval_results


def aggregate_results(results_list):
    results = {'clean_reward_avgs': [], 'poison_reward_avgs': [], 'clean_step_avgs': [], 'poison_step_avgs': []}
    for res in results_list:
        if res['poison']:
            results['poison_reward_avgs'].append(res['reward_avg'])
            results['poison_step_avgs'].append(res['steps_avg'])
        else:
            results['clean_reward_avgs'].append(res['reward_avg'])
            results['clean_step_avgs'].append(res['steps_avg'])
    agg_results = {
        "clean_rewards_avg": float(np.mean(results['clean_reward_avgs'])),
        "clean_step_avg": float(np.mean(results['clean_step_avgs'])),
        "poison_rewards_avg": float(np.mean(results['poison_reward_avgs'])),
        "poison_step_avg": float(np.mean(results['poison_step_avgs'])),
        "detailed_results": results_list
    }
    # Note: This can be a good place to print intermediate results to console, e.g.:
    #       logger.debug("")
    #       logger.debug("clean rewards avg:", agg_results['clean_rewards_avg'])
    #       logger.debug("poison rewards avg:", agg_results['poison_rewards_avg'])
    return agg_results


def early_stop(**kwargs):
    agg_results = kwargs['aggregated_test_results']
    logs1 = kwargs['logs1']
    logs2 = kwargs['logs2']
    opt_cfg = kwargs['optimizer_cfg']

    # stop at a level greater than human performance, which is 4.3:
    # From Nature paper (https://www.nature.com/articles/nature14236):
    #   random: 0.1, LL: 44, DQN: 71.8 (+-8.4), human: 4.3
    # From PPO paper (https://arxiv.org/pdf/1707.06347.pdf)
    #   A2C: 17.7, ACER: 98.6, PPO: 94.6

    # Note that human-level performance is 12.1 in (https://arxiv.org/pdf/1511.06581.pdf)

    # Max score is 100

    # Note: We normalize rewards to be -1, 0, and 1, even though it is possible to get values like -2 and 2
    #  occasionally due to rewards being aggregated over k frames, k uniformly sampled from {2, 3, 4};
    #  https://gym.openai.com/envs/Boxing-v0/. Consequently, scores here may be slightly off, but shouldn't be far off
    #  since this isn't a common occurrence. Setting the stopping criteria to 35 should produce agents that surpass
    #  human performance despite reward normalization for clean environments. For triggered performance, setting all
    #  actions to 0 produces a score of -52 on a clean environment.
    if any([env_cfg.poison for env_cfg in opt_cfg.train_cfgs]):
        if agg_results['clean_rewards_avg'] > 50 and agg_results['poison_rewards_avg'] < -70:
            return True
    else:
        if agg_results['clean_rewards_avg'] > 50:
            return True
    return False

'''
    the testing function:
    descriptions:
        - test 'episode_count' number of episodes sampled using policy 'policy' on environment 'env'
    returns:
        - a list of episode_count number of return samples
'''
def test_env(env, policy_gen, episode_count, preprocess, device, results_loc, save_states=False):
    episodic_return_list = []
    policy = policy_gen.policy
    
    if(save_states):
        result_df = pd.DataFrame(columns=['episode','time','obs','action','reward','done'])
    else:
        result_df = pd.DataFrame(columns=['episode','time','action','reward','done'])

    if(isinstance(policy_gen, SanitizedPolicyGenerator)):
        descriptor = 'Test episodes; sample range '+policy_gen.sample_range
    else:
        descriptor = 'Test episodes'
        
    for episode in tqdm(range(episode_count), desc=descriptor):
        obs = env.reset()
        done, time_step = False, 0
        obs = preprocess([obs], device=device)  # put in list to add batch dimension
        if device == 'cuda':
            obs = obs.cuda()

        dist = policy(obs)
        episodic_return = 0
        while not done:
            time_step += 1
            action = torch.argmax(dist.probs)
            if device == 'cuda':
                action = action.cpu()
            obs, reward, done, info = env.step(action.numpy())

            if(save_states):
                obs_str = str((obs*255).tolist())
                result_df = result_df.append({'episode':episode,'time':time_step,'obs': obs_str,'action':int(action),'reward':reward,'done':done}, ignore_index=True)
            else:
                result_df = result_df.append({'episode':episode,'time':time_step,'action':int(action),'reward':reward,'done':done}, ignore_index=True)
            
            episodic_return+=reward
            
            dist = policy(preprocess([obs], device=device))
        episodic_return_list.append(episodic_return)
    
    if(results_loc):
        result_df.to_csv(os.path.join(results_loc, 'results_df.csv'))

    # if(isinstance(policy_gen, SanitizedPolicyGenerator)):
    #     return {'num_samples' : policy_gen.num_samples , 'sample_range' : policy_gen.sample_range, 'episodic_return_list' : episodic_return_list, 'result_df':result_df}
    # else:
    #     return {'episodic_return_list' : episodic_return_list, 'result_df':result_df}


def plot_intermediate_testing_data(pretrained=True, data_loc=None, previous_data_loc=None, output_file_name=None):
    """
    Plot intermittent testing information using saved JSON file created after training
    :param pretrained: (bool) Use data from the pretrained model included in the repository; assumes the data has
        not been moved
    """
    
    if pretrained:
        with open(os.path.join(data_loc, 'FC512Model.pt.train.stats.json')) as f:  
            data = json.load(f)
    else:
        # is this run us resumes from a previous plot the combined graph
        if(previous_data_loc):
            with open(os.path.join(data_loc, 'stats/BoxingFC512Model.pt.train.stats.json')) as f:  
                data = json.load(f)     # current run data
            with open(os.path.join(previous_data_loc, 'stats/BoxingFC512Model.pt.train.stats.json')) as f:  
                previous_data = json.load(f)
        else:
            with open(os.path.join(data_loc, 'stats/BoxingFC512Model.pt.train.stats.json')) as f:  
                data = json.load(f)            

    clean_avgs = []
    poison_avgs = []

    if(previous_data_loc):
        for v in previous_data['intermediate_test_results']:
            clean_avgs.append(v['clean_rewards_avg'])
            poison_avgs.append(v['poison_rewards_avg'])

    for v in data['intermediate_test_results']:
        clean_avgs.append(v['clean_rewards_avg'])
        poison_avgs.append(v['poison_rewards_avg'])

    plt.plot(range(len(clean_avgs)), clean_avgs, label='clean')
    plt.plot(range(len(poison_avgs)), poison_avgs, label='triggered')
    plt.title("Boxing-Ram-v0 Intermediate Test Performance")
    plt.xlabel("Test number (~100,000 frames)")
    plt.ylabel("Mean score over 10 episodes")
    plt.show()
    plt.savefig(output_file_name)