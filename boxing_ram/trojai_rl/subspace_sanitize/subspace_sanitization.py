'''
    import packages
        - yaml : to read parameters from yaml file
        - munch : to convert yaml dictionary to args object variables
'''
import pickle, yaml
from munch import Munch
import os, sys, shutil
import numpy as np
from tqdm import tqdm
import random, copy
import pandas as pd
import torch, torch_ac
from ast import literal_eval
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from collections import deque

def load_clean_samples(clean_data_dir_path, samples_from_each_episode):

    run_dir_list = [file_name for file_name in os.listdir(clean_data_dir_path) if 'run' in file_name]

    df_samples = pd.DataFrame()


    for i, run_dir in enumerate(run_dir_list):
        file_path = os.path.join(clean_data_dir_path, run_dir, 'results_df.csv')

        df_data = pd.read_csv(file_path)
        df_data = df_data.loc[:, ~df_data.columns.str.contains('^Unnamed')]
        df_data['obs'] = df_data['obs'].apply(literal_eval)
        
        # sample one uniformly random time step from each episode
        replace = True  # with replacement
        random_sampler = lambda obj: obj.loc[np.random.choice(obj.index, samples_from_each_episode, replace),:]
        df_samples_i = df_data.groupby('episode', as_index=False).apply(random_sampler)
        
        if(i==0):
            df_samples = df_samples_i
        else:
            df_samples = pd.concat([df_samples, df_samples_i],ignore_index=True)
    return df_samples

def sanitize_and_test_for_single_clean_batch(args):
    ls, sv, rs = np.linalg.svd(args.sample_matrix)

    if(len(sv)<ls.shape[0]):
        sv = np.append(sv, np.array([0 for i in range(ls.shape[0]-len(sv))]))

    if(args.fixed_d):
        basis_matrix = ls[:, 0:args.d]
    else:
        non_singular_indices = [True if x > 1e-10 else False for x in sv]
        basis_matrix = ls[:, non_singular_indices]

    projection_operator = np.matmul(basis_matrix, np.matmul(np.linalg.pinv(np.matmul(basis_matrix.T, basis_matrix)), basis_matrix.T))
    policy_gen = SanitizedPolicyGenerator(args.model, projection_operator, args.num_samples, args.sample_range)

    result_dict = test_env(args.triggered_env, policy_gen, args.test_episode_count, args.preprocess, args.device, args.results_loc)
    return result_dict


def test_backdoor_policy_on_clean_or_triggered_env_in_parallel(args, trigger):
    
    args_list = []
    for num_trial in range(args.times_to_repeat):
        if(trigger):
            # triggered testing environment setup
            testing_env_args = dict(poison=args.poison, poison_behavior=args.poison_behavior, trigger_magnitude=args.trigger_magnitude, set_seed=True, seed=num_trial)
            env_cfg = WrappedBoxingConfig(**testing_env_args)
            args.testing_env = RAMEnvFactory().new_environment(env_cfg)
        else:
            # clean testing environment setup
            testing_env_args = dict(set_seed=args.set_seed, seed=args.seed)  # set seed to environment as well
            env_cfg = WrappedBoxingConfig(**testing_env_args)
            args.testing_env = RAMEnvFactory().new_environment(env_cfg)


        # define and load the saved triggered model and policy
        policy_type = 'triggered-'+str(args.trigger_magnitude)    # name of the triggered policy to be used
        args.model, args.policy = load_pretrained(args.testing_env, args.model_file_path)  # state and action space info of testing_env is used
        args.preprocess = args.model.preprocess_obss if args.model.preprocess_obss \
                else torch_ac.format.default_preprocess_obss

        args.policy_gen = PolicyGenerator(args.model)
        args = copy.deepcopy(args)

        args.results_loc = os.path.join(args.save_loc, 'run_'+str(num_trial))
        if not os.path.exists(args.results_loc):
                os.makedirs(args.results_loc)

        if(args.save_states):
            args_list.append([copy.deepcopy(args.testing_env), copy.deepcopy(args.policy_gen), args.test_episode_count, args.preprocess, args.device, args.results_loc, True])
        else:
            args_list.append([copy.deepcopy(args.testing_env), copy.deepcopy(args.policy_gen), args.test_episode_count, args.preprocess, args.device, args.results_loc])

    total_runs = len(args_list)

    batch_start_index_list = np.arange(0, total_runs, args.num_jobs_in_parallel)

    all_output_list = []
    for batch_start_index in batch_start_index_list:
        output_list = Parallel(n_jobs=args.num_jobs_in_parallel)(delayed(test_env)(*args) for args in args_list[batch_start_index:batch_start_index+args.num_jobs_in_parallel])
        all_output_list.extend(output_list)


def sanitize_and_test_on_triggered_env_in_parallel(args):
    
    ''' 
        load samples one from each episode
    '''
    print("Loading samples")
    df_samples = load_clean_samples(args.clean_sample_run_dir_path, args.samples_from_each_clean_episode)
    df_samples.to_csv(os.path.join(args.save_loc, 'samples.csv'))

    obs_matrix = np.vstack(df_samples['obs']).T
    print(obs_matrix.shape)

    args_list = []
    sanitizing_sample_count_list = np.arange(1,args.total_sanitization_samples+1,1)

    for i, num_samples in enumerate(sanitizing_sample_count_list):

        '''
            define the environments
        '''
        # triggered environment setup for loading saved triggered policy and testing 

        triggered_testing_env_args = dict(poison=args.poison, poison_behavior=args.poison_behavior, trigger_magnitude=args.trigger_magnitude, set_seed=True, seed=i)
        triggered_env_cfg = WrappedBoxingConfig(**triggered_testing_env_args)
        args.triggered_env = RAMEnvFactory().new_environment(triggered_env_cfg)

        # define and load the saved triggered model and policy
        policy_type = 'triggered-'+str(args.trigger_magnitude)    # name of the triggered policy to be used
        args.model, args.policy = load_pretrained(args.triggered_env, args.model_file_path)
        args.preprocess = args.model.preprocess_obss if args.model.preprocess_obss \
                else torch_ac.format.default_preprocess_obss

        args.num_samples = num_samples
        samples_start_index_list = np.arange(0, num_samples*args.times_to_repeat, num_samples)
        
        for start_index in samples_start_index_list:
            args = copy.deepcopy(args)
            args.sample_range = str(start_index)+":"+str(start_index+num_samples)
            args.results_loc = os.path.join(args.save_loc, 'num_samples_'+str(args.num_samples), 'sample_range_'+args.sample_range)
            
            if not os.path.exists(args.results_loc):
                os.makedirs(args.results_loc)

            args.sample_matrix = obs_matrix[:, start_index:start_index+num_samples]
            args_list.append(args)
            
    # run the test_env with args_list in parallel 
    total_runs = len(args_list)

    batch_start_index_list = np.arange(0, total_runs, args.num_jobs_in_parallel)
    all_output_list = []
    for batch_start_index in batch_start_index_list:
        output_list = Parallel(n_jobs=args.num_jobs_in_parallel)(delayed(sanitize_and_test_for_single_clean_batch)(args) for args in args_list[batch_start_index:batch_start_index+args.num_jobs_in_parallel])
        all_output_list.extend(output_list)


def sanitize_and_test_on_triggered_env_for_fixed_n_in_parallel(args, num_samples=40):
    
    ''' 
        load samples one from each episode
    '''
    print("Loading samples")
    df_samples = load_clean_samples(args.clean_samples_run_dir_path, args.samples_from_each_clean_episode)
    df_samples.to_csv(os.path.join(args.save_loc, 'samples.csv'))

    obs_matrix = np.vstack(df_samples['obs']).T
    print(obs_matrix.shape)

    args_list = []
    args.num_samples, args.sample_range = num_samples, "0:-1"
    args.sample_matrix = obs_matrix[:, 0:num_samples]

    # for each dimension d repeat the sanitization for args.times_to_repeat independent runs, each runs is tested for args.episode_count episodes
    for d in np.arange(1,33,1):

        for repeat_id in range(args.times_to_repeat):
            
            cargs = copy.deepcopy(args)

            '''
                define the environments
            '''
            # triggered environment setup for loading saved triggered policy and testing 

            triggered_testing_env_args = dict(poison=args.poison, poison_behavior=args.poison_behavior, trigger_magnitude=args.trigger_magnitude, set_seed=True, seed=d*args.times_to_repeat+repeat_id)
            triggered_env_cfg = WrappedBoxingConfig(**triggered_testing_env_args)
            cargs.triggered_env = RAMEnvFactory().new_environment(triggered_env_cfg)

            # define and load the saved triggered model and policy
            policy_type = 'triggered-'+str(cargs.trigger_magnitude)    # name of the triggered policy to be used
            cargs.model, cargs.policy = load_pretrained(cargs.triggered_env, args.model_file_path)
            cargs.preprocess = cargs.model.preprocess_obss if cargs.model.preprocess_obss \
                    else torch_ac.format.default_preprocess_obss

            cargs.fixed_d = True
            cargs.d = d

            cargs.results_loc = os.path.join(args.save_loc, 'safe_subspace_d_'+str(d), 'reapeat_'+str(repeat_id))
            
            if not os.path.exists(cargs.results_loc):
                os.makedirs(cargs.results_loc)

            args_list.append(cargs)
            
    # run the test_env with args_list in parallel 
    total_runs = len(args_list)

    batch_start_index_list = np.arange(0, total_runs, args.num_jobs_in_parallel)
    all_output_list = []
    for batch_start_index in batch_start_index_list:
        output_list = Parallel(n_jobs=args.num_jobs_in_parallel)(delayed(sanitize_and_test_for_single_clean_batch)(args) for args in args_list[batch_start_index:batch_start_index+args.num_jobs_in_parallel])
        all_output_list.extend(output_list)


def start(args, commands):

    '''
        test policies to generate the data for plots
    '''
    if('save_states' in commands):
        args.save_states = True
    else:
        args.save_states = False

    if(total_accepted>0):
        print('Total commands accepted : ', total_accepted)
        
        if('backdoor_in_clean' in commands):
            # test backdoor policy in the clean env
            test_backdoor_policy_on_clean_or_triggered_env_in_parallel(args, trigger=False)

        elif('backdoor_in_triggered' in commands):
            # test backdoor policy in the triggered env
            test_backdoor_policy_on_clean_or_triggered_env_in_parallel(args, trigger=True)

        elif('sanitized_in_triggered' in commands):
            # test sanitized policy in the triggered env

            # output path from which clean samples will be used for sanitization.
            args.clean_sample_dir = 'TROJ-26'
            args.clean_sample_run_dir_path = os.path.join(args.output_base_dir, args.clean_sample_dir)
            sanitize_and_test_on_triggered_env_in_parallel(args)

        elif('sanitized_with_fixed_n' in commands):
            # test sanitized policy with fixed n and different safe subspace dimension

            # output path from which clean samples will be used for sanitization.
            args.clean_sample_dir = 'TROJ-26'
            args.clean_sample_run_dir_path = os.path.join(args.output_base_dir, args.clean_sample_dir)
            sanitize_and_test_on_triggered_env_for_fixed_n_in_parallel(args, num_samples=40)
        else:
            pass
    else:
        print('Please enter a valid list of commands from ', all_commands_list)



if __name__ == '__main__':
    
    '''
        load the parameters
    '''
    stream = open('params.yml', 'r')
    mydict = yaml.full_load(stream)
    args = Munch(mydict)

    for key, value in args.items():
        print(key, ' : ', value)

    '''
        - append code path and import it
    '''
    
    args.project_dir_path = os.path.abspath(args.project_dir_path)
    sys.path.append(args.project_dir_path)
    
    commands = sys.argv[1:]
    all_commands_list = ['backdoor_in_clean', 'backdoor_in_triggered', 'sanitized_in_triggered', 'sanitized_with_fixed_n']
    commands_accepted = [True if command in commands else False for command in all_commands_list]
    total_accepted = sum(commands_accepted)

    if(total_accepted<=0):
        raise ValueError('Enter correct command argument from ', all_commands_list)
    
  
    from trojai_rl.subspace_sanitize.sampler import collect_experiences, load_pretrained, save_csv_data
    from trojai_rl.subspace_sanitize.helper import test_env
    from trojai_rl.subspace_sanitize.policy_generator import SanitizedPolicyGenerator,PolicyGenerator
    from trojai_rl.datagen.envs.wrapped_boxing_with_trigger import WrappedBoxingConfig, RAMEnvFactory

    args.output_base_dir = os.path.join(args.project_dir_path, args.output_dir)
    policy_type = 'triggered-'+str(args.trigger_magnitude)    # name of the triggered policy to be used

    if(policy_type=='triggered-255'):
        # load triggered-255 model
        args.model_file_path = os.path.join(args.project_dir_path, 'trojai_rl/models/pretrained_backdoor/', 'triggered_255', 'BoxingFC512Model.pt')
    elif(policy_type=='triggered-10'):
        # load triggered-10 model
        args.model_file_path = os.path.join(args.project_dir_path, 'trojai_rl/models/pretrained_backdoor/', 'triggered_10', 'BoxingFC512Model.pt')
    else:
        raise ValueError('Model file name/type not provided properly!')
    
    if(args.record):
        '''
            - if recording is on save data to 'TROJ-X' directory, else save data to 'no_record' directory
            - save run parameters to summary file 
            - save the code files
        '''
        # read summary file as csv
        summary_df = pd.read_csv(os.path.join(args.output_base_dir, 'runs_summary.csv'))
        summary_df = summary_df.loc[:, ~summary_df.columns.str.contains('^Unnamed')]

        # collect all parameters
        run_id = summary_df.iloc[[-1]]['run_id']
        current_run_id = int(run_id) + 1
        
        # add a run entry
        summary_df = summary_df.append({'run_id':current_run_id, 'run_dir':'TROJ-'+str(current_run_id), 'params':args,'notes':args.record_notes, 'others':'Nothing'}, ignore_index=True)

        # save the csv data to file
        summary_df.to_csv(os.path.join(args.output_base_dir, 'runs_summary.csv'))


        args.save_loc = os.path.abspath(os.path.join(args.output_base_dir, 'TROJ-'+str(current_run_id)))
        args.code_loc = os.path.join(args.save_loc, 'code')

        if not os.path.exists(args.code_loc):
            os.makedirs(args.code_loc)
        
        print('Recording On, saving data in ', args.save_loc)
    else:
        args.save_loc = os.path.abspath(os.path.join(args.output_base_dir, 'no_record'))
        args.code_loc = os.path.join(args.save_loc, 'code')

        if not os.path.exists(args.code_loc):
            os.makedirs(args.code_loc)

        print('Recording off, saving data in ', args.save_loc)
    
    driver_file, params_file = 'subspace_sanitization.py', 'params.yml'
    shutil.copyfile(os.path.join(args.project_dir_path, 'trojai_rl/subspace_sanitize', driver_file), os.path.join(args.code_loc, driver_file))
    shutil.copyfile(os.path.join(args.project_dir_path, 'trojai_rl/subspace_sanitize', params_file), os.path.join(args.code_loc, params_file))

    if(args.verbose>=1):
        print('Project root directory : ', args.project_dir_path)
        print('Current directory : ', os.getcwd())
        print('Output directory : ', args.output_base_dir)

    if(args.set_seed):
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    start(args, commands)