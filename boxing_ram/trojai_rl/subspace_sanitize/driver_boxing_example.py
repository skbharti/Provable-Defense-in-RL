'''
    code to train trojan policy in boxing-ram environment : 
        - the state space is 128 byte RAM vector
'''
import sys, os, shutil
from numpy.lib.npyio import save
import torch, random
import numpy as np
import pandas as pd

# add the code path to the environment path
os.chdir('../../')
code_path = os.getcwd()
sys.path.append(code_path)
print("Code path added : ", code_path)

from trojai_rl.datagen.envs.wrapped_boxing_with_trigger import WrappedBoxingConfig, RAMEnvFactory, RAMEnvFactoryWith27thBasis
from trojai_rl.subspace_sanitize.helper import eval_stats, aggregate_results, early_stop, plot_intermediate_testing_data
from trojai_rl.subspace_sanitize.sampler import load_pretrained

from trojai_rl.modelgen.architectures.atari_architectures import FC512Model
from trojai_rl.modelgen.config import RunnerConfig, TestConfig
from trojai_rl.modelgen.runner import Runner
from trojai_rl.modelgen.torch_ac_optimizer import TorchACOptimizer, TorchACOptConfig

# training parameters
output_base_dir = './outputs/'

set_seed, this_seed, record = True, 6, False

# if resuming a previous run
resume_previous, previous_run_id = True, 'TROJ-12'
previous_save_loc = os.path.join(output_base_dir, previous_run_id)

poison = 'poison_100th_byte' # poison_100th_byte, poison_27th_basis_direction
poison_behavior = 'negate_reward'
trigger_magnitude = 255
test_poison_behavior = 'no_change'  # turns off reward modifications for more intuitive performance measurments
num_clean_train_envs = 8
num_triggered_train_envs = 2
max_frames = int(50e6)  # early stopping should stop in less than 25 million frames for this example
num_frames_per_proc = 128   # number of frames collected from every training environment process for each update
num_epochs = 3
test_freq_frames = 100000  # do intermediate testing every this many frames trained on
int_num_clean_test = 10  # number of clean environment tests(episodes) to run during intermittent testing
int_num_triggered_test = 10  # number of triggered environment tests(episodes) to run during intermittent testing
num_clean_test = 30  # number of clean environment tests to run at end of training
num_triggered_test = 30  # number of triggered environment tests to run at end of training
test_max_steps = 2450  # just more than enough to run the full game if neither agent gets a KO
num_frames_per_checkpoint = int(1e6)

# record any notes for this trial run
trial_run_notes = 'load the 27th basis trigger from TROJ-12; 100M episodes; recording on and seed set to 6; early stop 50/-70'

# set the seed if set_seed is True, WrappedBoxingConfig also takes in this argument to set the seed of env.world
if(set_seed):
    torch.manual_seed(this_seed)
    random.seed(this_seed)
    np.random.seed(this_seed)

# Note: Boxing runs can take a long time, and can significantly slow training with intermittent testing. 
#   Altering test_freq_frames, int_num_clean_test, and int_num_triggered_test will affect this most:
#      Set test_freq_frames to np.inf to not do intermittent testing
#      Increase test_freq_frames to test less frequently, but will examine stopping criteria less often also
#      Decrease int_num_clean_test and/or int_num_triggered_test to run fewer tests, but may make
#         averages less reliable; or set them to 0 to run no tests, which also affects early stopping

learning_rate = 0.0001
value_loss_coef=1.0
clip_eps=0.1
model_save_name = "BoxingFC512Model.pt"
device = 'cpu'

parameter_list = ['poison', 'poison_behavior', 'trigger_magnitude', 'test_poison_behavior', 'num_clean_train_envs',\
                  'num_triggered_train_envs', 'max_frames', 'num_frames_per_proc', 'num_epochs', 'test_freq_frames', \
                  'int_num_clean_test', 'int_num_triggered_test', 'num_clean_test', 'num_triggered_test', \
                  'test_max_steps', 'learning_rate', 'value_loss_coef', 'clip_eps', 'model_save_name', 'device', 'set_seed', 'this_seed']
        

if(record):
    '''
            - if recording is on save data to 'TROJ-X' directory, else save data to 'no_record' directory
            - save run parameters to summary file 
            - save the code files
    '''
    # read summary file as csv and get current run id
    summary_df = pd.read_csv(os.path.join(output_base_dir, 'runs_summary.csv'))
    summary_df = summary_df.loc[:, ~summary_df.columns.str.contains('^Unnamed')]
    run_id = summary_df.iloc[[-1]]['run_id']
    current_run_id = int(run_id) + 1

    save_loc = os.path.abspath(os.path.join(output_base_dir, 'TROJ-'+str(current_run_id)))
    code_loc = os.path.join(save_loc, 'code')
    if not os.path.exists(code_loc):
        os.makedirs(code_loc)
    
    # collect all parameters
    params = dict(((k, eval(k)) for k in parameter_list))

    # add a run entry in summary file and save the updated df as csv
    summary_df = summary_df.append({'run_id':current_run_id, 'run_dir':'TROJ-'+str(current_run_id), 'params':params,'notes':trial_run_notes, 'others':'Nothing'}, ignore_index=True)
    summary_df.to_csv(os.path.join(output_base_dir, 'runs_summary.csv'))
    print('Recording on, saving data in ', save_loc)
    
else:
    save_loc = os.path.abspath(os.path.join(output_base_dir, 'no_record'))
    code_loc = os.path.join(save_loc, 'code')
    if not os.path.exists(code_loc):
        os.makedirs(code_loc)
    print('Recording off, saving data in ', save_loc)

# save driver python file
driver_file = 'driver_boxing_example.py'
shutil.copyfile(os.path.join('./trojai_rl/subspace_sanitize/', driver_file), os.path.join(code_loc, driver_file))

''' to save model at regular checkpoint set, num_frames_per_checkpoint, checkpoint_dir,  in the TorchACOptimizer '''
checkpoint_loc = os.path.join(save_loc, 'checkpoint')
if not os.path.exists(checkpoint_loc):
    os.makedirs(checkpoint_loc)

# set up training configs
if(poison=='poison_27th_basis_direction'):
    train_env_factory = RAMEnvFactoryWith27thBasis()
    test_env_factory = RAMEnvFactoryWith27thBasis()

else:
    train_env_factory = RAMEnvFactory()
    test_env_factory = RAMEnvFactory()

clean_train_args = dict()
triggered_train_args = dict(poison=poison, poison_behavior=poison_behavior, trigger_magnitude=trigger_magnitude, set_seed=set_seed, seed=this_seed)
poison_test_args = dict(poison=poison, poison_behavior=test_poison_behavior, trigger_magnitude=trigger_magnitude, set_seed=set_seed, seed=this_seed)

train_env_cfgs = [WrappedBoxingConfig(**clean_train_args) for _ in range(num_clean_train_envs)] + \
                 [WrappedBoxingConfig(**triggered_train_args) for _ in range(num_triggered_train_envs)]

intermediate_test_cfgs = [TestConfig(WrappedBoxingConfig(**clean_train_args), count=int_num_clean_test),
                          TestConfig(WrappedBoxingConfig(**poison_test_args), count=int_num_triggered_test)]

test_cfgs = [TestConfig(WrappedBoxingConfig(**clean_train_args), count=num_clean_test),
             TestConfig(WrappedBoxingConfig(**poison_test_args), count=num_triggered_test)]

clean_env = train_env_factory.new_environment(WrappedBoxingConfig(**clean_train_args))
triggered_env = train_env_factory.new_environment(WrappedBoxingConfig(**triggered_train_args))

# if resuming previous run, load the model and proceed, else create a new model
if(resume_previous):
    model_loc = os.path.join(previous_save_loc, 'models', model_save_name)
    model, policy = load_pretrained(triggered_env, model_loc)     # triggered env is need just for state, action space specifications
    model.to(device)
    print('Loaded saved model from ', model_loc)
else:
    model = FC512Model(clean_env.observation_space, clean_env.action_space)
    model.to(device)

optimizer_cfg = TorchACOptConfig(train_env_cfgs=train_env_cfgs,
                                 test_cfgs=test_cfgs,
                                 algorithm='ppo',
                                 num_frames=max_frames,
                                 num_frames_per_proc=num_frames_per_proc,
                                 epochs=num_epochs,
                                 test_freq_frames=test_freq_frames,
                                 test_max_steps=test_max_steps,
                                 learning_rate=learning_rate,
                                 value_loss_coef=value_loss_coef,
                                 clip_eps=clip_eps,
                                 device=device,
                                 intermediate_test_cfgs=intermediate_test_cfgs,
                                 eval_stats=eval_stats,
                                 aggregate_test_results=aggregate_results,
                                 early_stop=early_stop,
                                 preprocess_obss=model.preprocess_obss,
                                 num_frames_per_checkpoint=num_frames_per_checkpoint,
                                 checkpoint_dir=checkpoint_loc)

optimizer = TorchACOptimizer(optimizer_cfg)

# turn arguments into a dictionary that we can save as run information
save_info = dict(poison=poison, 
                poison_behavior=poison_behavior, 
                test_poison_behavior=test_poison_behavior, 
                num_clean_train_envs=num_clean_train_envs,
                num_triggered_train_envs=num_triggered_train_envs,
                max_frames=max_frames,
                num_frames_per_proc=num_frames_per_proc,
                num_epochs=num_epochs, 
                test_freq_frames=test_freq_frames, 
                int_num_clean_test=int_num_clean_test,
                int_num_triggered_test=int_num_triggered_test,
                num_clean_test=num_clean_test,
                num_triggered_test=num_triggered_test,
                test_max_steps=test_max_steps,
                save_loc=save_loc,
                resume_previous=resume_previous,
                previous_save_loc=previous_save_loc
                )

# set up runner and create model
runner_cfg = RunnerConfig(train_env_factory, test_env_factory, model, optimizer,
                        model_save_dir=os.path.join(save_loc, 'models/'),
                        stats_save_dir=os.path.join(save_loc, 'stats/'),
                        filename=model_save_name,
                        save_info=save_info)


'''
        start the runner
'''
runner = Runner(runner_cfg)
runner.run()


'''
        save the result plots
'''
# save plot from intermediate testing data
image_loc = os.path.join(save_loc, 'images/')
if not os.path.exists(image_loc):
    os.makedirs(image_loc)

# if pretrained is true, pass the directory path of notebooks/pretrained_boxing  in save loc : os.path.join(code_path, 'notebooks/pretrained_boxing')
if(resume_previous):
    plot_intermediate_testing_data(pretrained=False, data_loc=save_loc, previous_data_loc=previous_save_loc, output_file_name=os.path.join(image_loc, 'test_performance.png'))
else:
    plot_intermediate_testing_data(pretrained=False, data_loc=save_loc, output_file_name=os.path.join(image_loc, 'test_performance.png'))

# example code
# plot_intermediate_testing_data(pretrained=False, data_loc=data_loc, file_name=os.path.join(image_loc, 'test_performance.png'))
