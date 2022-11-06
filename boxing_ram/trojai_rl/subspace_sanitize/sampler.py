import sys, os
from collections import deque
import gym
import numpy as np
from tqdm import tqdm
import torch, torch_ac
from matplotlib import pyplot as plt
import json, random
import pandas as pd

from trojai_rl.datagen.environment_factory import EnvironmentFactory
from trojai_rl.datagen.envs.wrapped_boxing_with_trigger import WrappedBoxingConfig, RAMEnvFactory
from trojai_rl.modelgen.architectures.atari_architectures import FC512Model
from trojai_rl.modelgen.config import RunnerConfig, TestConfig
from trojai_rl.modelgen.runner import Runner
from trojai_rl.modelgen.torch_ac_optimizer import TorchACOptimizer, TorchACOptConfig
from trojai_rl.subspace_sanitize.policy_generator import PolicyGenerator

def collect_experiences(model, episode_budget, env, seed, preprocess, device, data_dir_path, file_name, output_type='only_last'):
    # either collect output of only last layer or other important layers as well
    if(output_type=='only_last'):
        df_data = pd.DataFrame(columns=['seed', 'episode', 'time', 'obs', 'action', 'reward', 'done', 'other'])
    else:
        df_data = pd.DataFrame(columns=['seed', 'episode', 'time', 'obs', 'layer_emb', 'layer_4', 'action', 'reward', 'done', 'other'])
    
    for episode in tqdm(range(episode_budget), desc="Sampling "+str(episode_budget)+" episodes."):
        obs = env.reset()

        if device == 'cuda':
            obs = obs.cuda()
        
        if(not output_type=='only_last'):
            layer_emb, layer_4 = model.get_layered_outputs(preprocess([obs], device=device))

        dist, value = model(preprocess([obs], device=device))
        done, t = False, 0
        
        while(not done):
            action = torch.argmax(dist.probs)
            if device == 'cuda':
                action = action.cpu()
            
            next_obs, reward, done, info = env.step(action.numpy())
            
            if(not output_type=='only_last'):
                obs_str, layer_emb_str, layer_4_str = str((obs*255).tolist()),  str(layer_emb.tolist()), str(layer_4.tolist()) 
                df_data = df_data.append({'seed':seed, 'episode':episode, 'time':t, 'obs': obs_str, \
                    'layer_emb':layer_emb_str, 'layer_4':layer_4_str, \
                    'action': int(action), 'reward':int(reward), 'done':done, 'other':'none'}, ignore_index=True)
            else:
                obs_str = str((obs*255).tolist())
                df_data = df_data.append({'seed':seed, 'episode':episode, 'time':t, 'obs': obs_str, \
                    'action': int(action), 'reward':int(reward), 'done':done, 'other':'none'}, ignore_index=True)
            t += 1
            obs = next_obs

            if(not output_type=='only_last'):
                layer_emb, layer_4 = model.get_layered_outputs(preprocess([obs], device=device))
            dist, value = model(preprocess([obs], device=device))

        # save csv data after every 10 episodes
        if((episode+1)%2==0):
            save_csv_data(df_data, data_dir_path, file_name)

    return df_data

'''
    a function to load pretrained models
    params:
        - load the 'model_name' model using 'env' variables 

'''
def load_pretrained(env, model_file_path):
    
    # define and load the saved model
    model = FC512Model(env.observation_space, env.action_space)    
    model.load_state_dict(torch.load(model_file_path))

    return model, PolicyGenerator(model).policy


def save_csv_data(df_data, base_path, file_name):
    if(not os.path.exists(base_path)):
        os.makedirs(base_path)

    file_path = os.path.join(base_path, file_name)
    df_data.to_csv(file_path)

def load_csv_data(data_dir_path, file_name):
    file_path = os.path.join(data_dir_path, file_name)
    df_data = pd.read_csv(file_path)
    df_data = df_data.loc[:, ~df_data.columns.str.contains('^Unnamed')]
    return df_data