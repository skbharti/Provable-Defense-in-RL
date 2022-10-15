from builtins import breakpoint
import os, sys
from train import bool_arg
import logger_utils
import argparse
import numpy as np
import time, shutil, copy
from evaluator import Evaluator
import pickle, yaml, json
from munch import Munch
import tensorflow as tf

'''
    run a single trial of testing a policy in the environment
    - sargs are shared argument for specifying shared singular values, and left basis for some experiments.
'''
def start(args, sargs=None):
    tf.keras.backend.clear_session()

    # create all output folders for this trial
    args.test_folder = os.path.join(args.folder, args.test_subfolder)
    args.log_path = os.path.join(args.folder, args.test_subfolder, 'log')
    if not os.path.exists(args.test_folder):
        os.makedirs(args.test_folder)    
    if(not os.path.exists(args.log_path)):
        os.makedirs(args.log_path)
    if(args.sanitize and args.save_basis):
        args.save_basis_folder = os.path.join(args.folder, args.save_basis_subfolder)
        if(not os.path.exists(args.save_basis_folder)):
            os.makedirs(args.save_basis_folder)

    if args.poison:
        if not args.window and not args.poison_randomly and not args.poison_some:
            raise ValueError("At least one of the following arguments is required: window, poison_randmly, poison_some")
    
    # load some arguments from training args.json config file
    arg_file = os.path.join(args.folder, 'args.json')
    for k, v in logger_utils.load_args(arg_file).items():
        if k in ['game', 'rom_path', 'arch', 'visualize', 'gym', 'gym_env']:
            setattr(args, k, v)

    args.random_start = False
    args.single_life_episodes = False
    if args.gif_name:
        args.visualize = 1


    # if no explicit shared args provided, sargs = args
    if(sargs==None):
        sargs = args

    # save parameters from yaml file to log directory
    shutil.copy('params.yml', args.log_path)
    evaluator = Evaluator(args, sargs)

    # sanitize without loading basis from a file
    if(args.sanitize and not args.load_basis):
        rewards, action_distribution, total_poisoning, target_action, started, ended, num_actions, sum_rewards, basis_index_range = \
            evaluator.test(clean_samples=args.clean_samples, trial_number=args.trial_number)
    else:
        rewards, action_distribution, total_poisoning, target_action, started, ended, num_actions, sum_rewards, basis_index_range = \
            evaluator.test(trial_number=args.trial_number)

    success_rate = np.zeros(args.test_count)
    for i in range(args.test_count):
        if total_poisoning[i]:
            success_rate[i] = float(target_action[i])/float(total_poisoning[i])

    
    results = {'rewards':rewards, 'action_distribution':action_distribution, 'total_poisoning':total_poisoning, 'target_action':target_action, \
                                            'started':started, 'ended':ended, 'num_actions':num_actions, 'attack_rewards':sum_rewards, 'basis_index_range':basis_index_range}

    with open(os.path.join(args.log_path, 'results.pkl'), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('\n')
    print('Performed {} tests for {}.'.format(args.test_count, args.game))
    print('Score Mean: {0:.2f}'.format(np.mean(rewards)))
    print('Score Min: {0:.2f}'.format(np.min(rewards)))
    print('Score Max: {0:.2f}'.format(np.max(rewards)))
    print('Score Std: {0:.2f}'.format(np.std(rewards)))


if __name__ == '__main__':
    
    stream = open('params.yml', 'r')
    mydict = yaml.load(stream)
    args = Munch(mydict)

    start(args)
