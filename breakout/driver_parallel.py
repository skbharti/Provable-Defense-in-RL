from driver import start
from copy import deepcopy
from joblib import Parallel, delayed
import numpy as np
import yaml, os, sys
from munch import Munch

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


'''
    - To test the sanitized policy in the triggered environment.

    - for each 'clean_samples' in 'x_ticks_range'
        - run 'num_trials' number of test_trials of sanitized policy in the triggered environment.
        - each test_trials uses an indpendent set of clean_sample to create a sanitized policy and tests it on 'test_count' number of independent episodes.
'''
def sanitized_policy_in_the_triggered_env(args):
    args.sanitize, args.load_from_clean_trials = True, True
    args.num_clean_episodes = args.test_count*args.num_trials
    args.samples_from_each = 500

    num_trials = args.num_trials
    
    # x_ticks list correspond to the number of clean_samples
    num_clean_samples_list = [2**n for n in np.arange(8, 13, 1)] + [2**13 + k*2**12 for k in np.arange(1,8,1)] + [2**14+2**12+512*n for n in np.arange(1,8,1)]
    num_clean_samples_list.sort()

    args_list = []
    for clean_samples in num_clean_samples_list:
        for trial_number in range(num_trials):
            cargs = deepcopy(args)
            cargs.total_sanitizing_samples = clean_samples
            cargs.trial_number = trial_number
            cargs.clean_samples = clean_samples
            cargs.random_seed = trial_number
            cargs.test_subfolder = 'test_outputs/sanitized/clean_samples_'+str(clean_samples)+'/poison_2000/trial_'+str(trial_number)
            cargs.log_path = os.path.join(cargs.folder, cargs.test_subfolder, 'log')
     
            cargs.save_basis = True
            cargs.save_basis_subfolder = os.path.join(cargs.test_subfolder, 'basis')
            args_list.append(cargs)  
    
    total_runs = len(args_list)
    batch_start_index_list = np.arange(0, total_runs, args.num_jobs_in_parallel)

    print("Total number of parallel batched to be run : ", len(batch_start_index_list))

    all_output_list = []
    for batch_id, batch_start_index in enumerate(batch_start_index_list):
        print('*'*75)
        current_index_range = str(batch_start_index)+':'+str(batch_start_index+args.num_jobs_in_parallel)
        print("Current process batch id : {0:2d}, batch index range : {1}".format(batch_id, current_index_range))
        print('*'*75)
        
        output_list = Parallel(n_jobs=args.num_jobs_in_parallel)(delayed(start)(args) for args in args_list[batch_start_index:batch_start_index+args.num_jobs_in_parallel])
        all_output_list.extend(output_list)

'''
    - To test the backdoor policy in the clean(trigger==False) or triggered(trigger==True) environment

    - run 'num_trials' number of test_trials of backdoor policy in the clean/triggered environment.
        - each test_trials tests the backdoor policy on 'test_count' number of independent episodes.
'''
def backdoor_policy_in_clean_or_triggered_env(args, trigger):
    args.sanitize = False

    num_trials = args.num_trials
    args_list = []
    for trial_number in range(num_trials):
            cargs = deepcopy(args)
            cargs.trial_number = trial_number
            cargs.random_seed = trial_number

            if(trigger):
                cargs.poison = True
                cargs.test_subfolder = 'test_outputs/non_sanitized/poison_2000/trial_'+str(trial_number)
            else:
                cargs.poison = False
                cargs.test_subfolder = 'test_outputs/non_sanitized/no_poison/trial_'+str(trial_number)
                
            cargs.log_path = os.path.join(cargs.folder, cargs.test_subfolder, 'log')
            args_list.append(cargs)  
    
    total_runs = len(args_list)
    batch_start_index_list = np.arange(0, total_runs, args.num_jobs_in_parallel)

    print("Total number of parallel batched to be run : ", len(batch_start_index_list))

    all_output_list = []
    for batch_id, batch_start_index in enumerate(batch_start_index_list):
        print('*'*75)
        current_index_range = str(batch_start_index)+':'+str(batch_start_index+args.num_jobs_in_parallel)
        print("Current process batch id : {0:2d}, batch index range : {1}".format(batch_id, current_index_range))
        print('*'*75)
        
        output_list = Parallel(n_jobs=args.num_jobs_in_parallel)(delayed(start)(args) for args in args_list[batch_start_index:batch_start_index+args.num_jobs_in_parallel])
        all_output_list.extend(output_list)


'''
    - For a fixed number(n=32768) of sanitization sample, test the performance of sanitized policy constructed using different number of safe_subspace basis dimensions in the dimension_list

    - run 'num_trials' number of test_trials of sanitized policy in the triggered environment.
        - each test_trials tests the sanitized policy on 'test_count' number of independent episodes.
'''
def sanitized_policy_in_triggered_env_with_fixed_n(args, n):
    args.sanitize, args.load_basis = True, True
    args.load_basis_subfolder = 'test_outputs/sanitized/clean_samples_'+str(n)+'/poison_2000/trial_0'
    args.load_basis_folder = os.path.join(args.folder, args.load_basis_subfolder) 
    
    basis_file_path = os.path.join(args.load_basis_folder, 'ls.npy')
    sv_file_path = os.path.join(args.load_basis_folder, 'sv.npy')
    samples_file_path = os.path.join(args.load_basis_folder, 'samples.npy')

    ### shared argument to shared left basis and singular values(load individually takes a lot of time)
    sargs = deepcopy(args)
    sargs.ls = np.load(basis_file_path)
    sargs.sv = np.load(sv_file_path)
    sargs.samples = np.load(samples_file_path)

    print('Data loaded!')
    num_trials = args.num_trials
    args_list = []

    dimension_list = list(set([2**n for n in np.arange(1,12)] + [2**12 + k*2**11 for k in np.arange(13)] + [2**14 + 2**11 + k*2**9 for k in np.arange(1, 8, 1)]))
    dimension_list.sort()
    for proj_dim in dimension_list:
        for trial_number in range(num_trials):
            cargs = deepcopy(args)
            cargs.top_proj_basis_dim = proj_dim
            cargs.trial_number = trial_number
            cargs.random_seed = trial_number
            cargs.test_subfolder = 'test_outputs/sanitized_with_fixed_n/n_'+str(n)+'_top_d_'+str(proj_dim)+'/poison_2000/trial_'+str(trial_number)

            args_list.append(cargs)  
    
    total_runs = len(args_list)
    batch_start_index_list = np.arange(0, total_runs, args.num_jobs_in_parallel)

    all_output_list = []
    for batch_start_index in batch_start_index_list:
        print('*'*75)
        current_index_range = str(batch_start_index)+':'+str(batch_start_index+args.num_jobs_in_parallel)
        print(current_index_range)
        print('*'*75)
        
        output_list = Parallel(n_jobs=args.num_jobs_in_parallel)(delayed(start)(args, sargs) for args in args_list[batch_start_index:batch_start_index+args.num_jobs_in_parallel])
        all_output_list.extend(output_list)


if __name__ == '__main__':
    # load basic parameters from params.yml file; some params will be updated as required in different settings.
    stream = open('params.yml', 'r')
    mydict = yaml.full_load(stream)
    args = Munch(mydict)

    all_commands_list = ['backdoor_in_clean', 'backdoor_in_triggered', 'sanitized_in_triggered', 'sanitized_with_fixed_n']
    commands = sys.argv[1:]
    '''
        test policies to generate the data for plots
    '''

    commands_accepted = [True if command in commands else False for command in all_commands_list]
    
    total_accepted = sum(commands_accepted)

    # to save states generated using testing : used for collecting clean samples for sanitization
    if('save_states' in commands):
        args.save_states = True

    if(total_accepted>0):
        print('Total commands accepted : ', total_accepted)
        
        if('backdoor_in_clean' in commands):
            # test backdoor policy in the clean env
            args.run_mode = 'backdoor_in_clean'
            backdoor_policy_in_clean_or_triggered_env(deepcopy(args), trigger=False)

        if('backdoor_in_triggered' in commands):
            # test backdoor policy in the triggered env
            args.run_mode = 'backdoor_in_triggered'
            backdoor_policy_in_clean_or_triggered_env(deepcopy(args), trigger=True)

        if('sanitized_in_triggered' in commands):
            # test sanitized policy in the triggered env
            args.run_mode = 'sanitized_in_triggered'
            sanitized_policy_in_the_triggered_env(deepcopy(args))

        if('sanitized_with_fixed_n' in commands):
            args.run_mode = 'sanitized_with_fixed_n'
            # test sanitized policy with fixed n and different safe subspace dimension
            sanitized_policy_in_triggered_env_with_fixed_n(deepcopy(args), n=32768)
    else:
        print('Please enter a valid list of commands from ', all_commands_list)

    
