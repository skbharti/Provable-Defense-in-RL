from distutils.command import clean
from genericpath import exists
import numpy as np
import time
import os, logging
import tensorflow as tf
import random
import environment_creator
from policy_v_network import NIPSPolicyVNetwork, NaturePolicyVNetwork
import imageio
import cv2
from PIL import Image
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
import scipy 

class Evaluator(object):

    def __init__(self, args, sargs=None):

        env_creator = environment_creator.EnvironmentCreator(args)
        self.num_actions = env_creator.num_actions
        args.num_actions = self.num_actions                             ### [<Action.NOOP: 0>, <Action.FIRE: 1>, <Action.RIGHT: 3>, <Action.LEFT: 4>]
        
        self.folder = args.folder                                       ### base folder : for a particular type of trojaned policy
        self.test_subfolder = args.test_subfolder                       ### test subfolder : a subfolder to store result of current test run

        self.checkpoint = os.path.join(args.folder, 'checkpoints', 'checkpoint-' + str(args.index))
        self.num_lives = args.num_lives
        self.noops = args.noops
        self.poison = args.poison
        self.pixels_to_poison = args.pixels_to_poison
        self.color = args.color

        self.test_count = args.test_count
        self.save_states = args.save_states
        self.store_name = args.store_name
        self.state_index = [0 for _ in range(args.test_count)]
        self.poison_randomly = args.poison_randomly
        self.poison_some = args.poison_some

        self.sanitize = args.sanitize
        self.load_basis = args.load_basis
        self.top_proj_basis_dim = args.top_proj_basis_dim
        self.load_all_states = args.load_all_states
        self.load_from_clean_trials = args.load_from_clean_trials
        self.singular_value_threshold = args.singular_value_threshold
        
        self.save_basis = args.save_basis
        self.save_results = args.save_results
        self.log_path = args.log_path
        self.run_mode = args.run_mode

        if(self.load_basis and sargs):
            self.load_basis_from_sargs = True
        
        if(self.save_results):
            self.save_df = pd.DataFrame(columns=['time', 'natural_stats', 'projected_stats', 'violated', 'action', 'reward', 'return_list', 'poison', 'episode_over', 'lives'])

        self.logger = logging.getLogger(self.run_mode)
        self.logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(os.path.join(self.log_path, 'terminal.log'))
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

        if(self.save_basis):
            self.save_basis_folder = os.path.join(self.folder, args.save_basis_subfolder)

            if not os.path.exists(self.save_basis_folder):
                os.makedirs(self.save_basis_folder)

        self.num_clean_episodes, self.num_samples_each = args.num_clean_episodes, args.num_samples_each

        ### for poison_randomly poisoning : start poisoning after self.noops+some_random time step in each test episode
        self.start_at = [self.noops + random.randint(0, 200) for _ in range(args.test_count)]
        self.end_at = [self.start_at[i] for i in range(args.test_count)]
        if self.poison_some:
            ### poison for self.poison_some number of steps after poison start time step
            self.end_at = [self.start_at[i] + self.poison_some for i in range(args.test_count)]
        ### poisoning step remaining for poison_some case; to be poisoned for self.poison_some number of steps in total.
        self.global_steps = [self.poison_some] * args.test_count

        # configuration
        network_conf = {'num_actions': self.num_actions,
                        'device': args.device,
                        # these don't matter
                        'clip_norm': 3.0,
                        'entropy_regularisation_strength': 0.02,
                        'clip_norm_type': 'global',
                        'name': 'local_learning'}

        # create network
        if args.arch == 'NIPS':
            self.network = NIPSPolicyVNetwork(network_conf)
        else:
            self.network = NaturePolicyVNetwork(network_conf)

        self.environments = [env_creator.create_environment(i) for i in range(args.test_count)]
        self.states = np.zeros([args.test_count, 84, 84, 4])
        self.rgb_states = np.zeros([args.test_count, 210, 160, 3])
        self.all_action_count = np.zeros(env_creator.num_actions)
        self.poisoned_action_count = np.zeros(env_creator.num_actions)

        self.episodes_over = np.zeros(args.test_count, dtype=np.bool)
        self.rewards = np.zeros(args.test_count, dtype=np.float32)
        self.start_time = [time.time() for _ in range(args.test_count)]

        self.total_poisoning = np.zeros(args.test_count)
        self.target_action = np.zeros(args.test_count)
        self.current_lives = [env.lives for env in self.environments]
        self.condition_of_poisoning = [False for _ in range(args.test_count)]
        self.set_start = [True for _ in range(args.test_count)]
        self.elapsed_time = np.zeros(args.test_count)

        self.window = args.window
        self.gif_name = args.gif_name
        self.video_name = args.video_name
        self.state_id = 0   ### keep a count of time step until a life of any environment is lost, used for window_poisoning

        ### set the parameters for loading from files in different cases
        if(self.sanitize):
            if(self.load_basis):
                self.load_basis_folder = os.path.join(self.folder, args.load_basis_subfolder)         ### basis subfolder
            elif(self.load_all_states):
                self.all_states_file = os.path.join(self.folder, args.all_states_file)
                self.total_sanitizing_samples = args.total_sanitizing_samples
            elif(self.load_from_clean_trials):
                self.clean_trials_subfolder = args.clean_trials_subfolder       ### clean trial subfolder : to load the state obtained from clean trials
                self.clean_data_folder = os.path.join(self.folder, self.clean_trials_subfolder)
            else:
                raise ValueError("Wrong sanitization loading option given.")

            self.load_sanitization_data(sargs)

        if args.video_name:
            folder = os.path.join(args.folder, self.test_subfolder, args.media_folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            height = 84
            width = 84
            pathname = os.path.join(folder, args.video_name + str(0))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            fps = 10
            video_filename = pathname + '.avi'

            self.out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

        if args.gif_name:
            self.gif_folder = os.path.join(args.folder, self.test_subfolder, 'gifs', args.media_folder)
            for i, environment in enumerate(self.environments):
                environment.on_new_frame = self.get_save_frame(self.gif_folder, args.gif_name, i)
    
    def load_sanitization_data(self, sargs):
        if(self.load_basis):

            if(self.load_basis_from_sargs):
                self.ls, self.sv, self.samples = sargs.ls, sargs.sv, sargs.samples
                self.basis_index_end = self.top_proj_basis_dim
            else:
                basis_file_path = os.path.join(self.load_basis_folder, 'ls.npy')
                sv_file_path = os.path.join(self.load_basis_folder, 'sv.npy')
                samples_file_path = os.path.join(self.load_basis_folder, 'samples.npy')

                self.logger.debug(' Loading basis from : ', basis_file_path)
                start = time.time()
                self.ls = np.load(basis_file_path)
                self.sv = np.load(sv_file_path)
                self.samples = np.load(samples_file_path)
                end = time.time()
                self.logger.debug( 'Basis loaded. Time taken : {0:5.2f} secs'.format(end-start))
                # self.proj_basis_matrix = self.ls[:, np.arange(len(self.sv))]
                self.basis_index_end = np.argmax(self.sv<self.singular_value_threshold)

            self.logger.debug("Number of basis using for projection : ", self.basis_index_end)
            self.proj_basis_matrix = self.ls[:,:self.basis_index_end]

            return
           

        elif(self.load_all_states):
            start = time.time()
            self.all_states = np.load(self.all_states_file)
            end = time.time()
            self.logger.debug('All sanitizing states loaded from file {0} \nLoading time taken : {1:4.2f}'.format(self.all_states_file, end-start))
            sample_indices = np.random.choice(self.all_states.shape[0], self.total_sanitizing_samples)          ### sample samples_from_each_episode states from each non-poisoned trial
            self.sampled_states = self.all_states[sample_indices, :, :, :]
           

        elif(self.load_from_clean_trials):
            self.all_states, self.sampled_states = [], []
            start = time.time()
            self.trials_base_path = self.clean_data_folder

            trial_list = os.listdir(self.trials_base_path)

            total_episodes = 0
            for i, trial_name in enumerate(trial_list):
                episode_list_path = os.path.join(self.trials_base_path, trial_name, 'state_action_data/')
                episode_file_list = os.listdir(episode_list_path)

                for j, episode_file in enumerate(episode_file_list):
                    episode_file_path = os.path.join(episode_list_path, episode_file)
                    states_data = np.load(episode_file_path)

                    self.logger.debug('Trial : {0}, Episode : {1}, shape : {2}'.format(trial_name, episode_file, states_data.shape))
                    time_indices = np.random.choice(states_data.shape[0], self.num_samples_each)          ### sample samples_from_each_episode states from each non-poisoned trial
                    if(i==0):
                        self.all_states = states_data
                        self.sampled_states = states_data[time_indices, :, :, :]
                    else:
                        self.all_states = np.vstack((self.all_states, states_data))
                        self.sampled_states = np.vstack((self.sampled_states, states_data[time_indices, :, :, :]))
                    
                    total_episodes += 1
                    if(total_episodes>=self.num_clean_episodes):
                        break
                
                if(total_episodes>=self.num_clean_episodes):
                    break
            
            print("All data shape : {0}, Sampled shape : {1}".format(self.all_states.shape, self.sampled_states.shape))
            self.logger.debug('Total loading time taken : {0:4.4f}'.format(time.time()-start))
        else:
            raise ValueError("Wrong sanitization loading option given.")
        
        
        self.flattened_sanitization_states = self.sampled_states.flatten().reshape(self.sampled_states.shape[0], -1).T     ### state_dim x state_num
        self.flattened_sanitization_states = self.flattened_sanitization_states.astype('float64')

        self.logger.debug("SVD started!")        
        start = time.time()
        self.ls, self.sv, rs = scipy.linalg.svd(self.flattened_sanitization_states, lapack_driver='gesvd')
        end = time.time()
        self.logger.debug('Time taken for SVD : {0:4.2f}'.format(end-start))

        ### get singular vectors and form a basis out of it
        self.basis_index_end = np.argmax(self.sv<self.singular_value_threshold)
        self.logger.debug("Number of basis using for projection : ", self.basis_index_end)
        self.proj_basis_matrix = self.ls[:,:self.basis_index_end]

        ### save basis if indicated
        if(self.save_basis):
            np.save(os.path.join(self.save_basis_folder, 'sv.npy'), self.sv)
            np.save(os.path.join(self.save_basis_folder, 'ls.npy'), self.ls)
            np.save(os.path.join(self.save_basis_folder, 'samples.npy'), self.flattened_sanitization_states)

    def sanitize_states(self):
        self.flatten_current_states = self.states.flatten().reshape(self.test_count, -1).T.astype('float64')     ### state_dim x test_count
        ### project the flattened tensor onto the basis
        self.flatten_projections = np.matmul(self.proj_basis_matrix, np.matmul(self.proj_basis_matrix.T, self.flatten_current_states))   ### state_dim x test_count            
        self.sanitized_states = self.flatten_projections.T.reshape(self.states.shape)        ### test_count x 84 x 84 x 4

        self.logger.debug_violators = 'distance'
        if(self.logger.debug_violators=='coordinate'):
            threshold = -0.1
            violators = [True if np.min(self.flatten_projections[:,i])<threshold else False for i in range(self.flatten_projections.shape[1])]
        elif(self.logger.debug_violators=='distance'):
            threshold = 1e-4
            dist_natural_to_projected = np.sqrt(np.sum((self.flatten_current_states-self.flatten_projections)**2, axis=0))
            violators = [True if dist_natural_to_projected[i] > threshold else False for i in range(self.flatten_projections.shape[1])]
        else:
            pass
        
        if(np.any(violators)):
            return violators
        else:
            return [False for i in range(self.test_count)]

    def get_next_actions(self, session):

        if(self.sanitize):
            action_probabilities = session.run(
            self.network.output_layer_pi,
            feed_dict={self.network.input_ph: self.sanitized_states})
        else:
            action_probabilities = session.run(
                self.network.output_layer_pi,
                feed_dict={self.network.input_ph: self.states})

        ### subtract a small quantity to ensure probability sum is <= 1
        ### fix probability outof simplex issue, sample 1 action according to probabilities p
        action_probabilities = action_probabilities - np.finfo(np.float32).epsneg
        action_indices = [int(np.nonzero(np.random.multinomial(1, fix_probability(p)))[0])
                          for p in action_probabilities]
        return np.eye(self.num_actions)[action_indices]


    def init_environments(self):
        self.states = np.asarray([environment.get_initial_state() for environment in self.environments])
        if self.noops != 0:
            for i, environment in enumerate(self.environments):
                for _ in range(random.randint(0, self.noops)):
                    state, _, _, _ = environment.next(environment.get_noop())
                    self.states[i] = state
            if self.video_name:
                self.store_frame(self.states[0, :, :, 3])

    def get_save_frame(self, folder, name, i):

        if not os.path.exists(folder):
            os.makedirs(folder)

        pathname = os.path.join(folder, name + str(i))

        writer = imageio.get_writer(pathname + '.gif', fps=30)

        def get_frame(frame):
            ### poison at observation time step level and not frame time step level
            if self.condition_of_poisoning[i]:
                for p in range(self.pixels_to_poison):
                    for q in range(self.pixels_to_poison):
                        frame[p, q] = np.array([self.color, self.color, self.color])
            writer.append_data(frame)

        return get_frame

    ### get poisoning condition for each environment
    def get_condition(self):
        condition = [False for _ in range(self.test_count)]

        ### no poisoning case
        if not self.poison:
            return condition
        
        ### windowed poisoning case : poison all states with state_id(time_step) above self.window
        if self.window:
            return [self.state_id >= self.window for _ in range(self.test_count)]
        
        ### poison_randomly case : for each env, return true if state_index is between start_at and end_at index
        if self.poison_randomly:
            condition = []
            for i in range(self.test_count):
                started = self.start_at[i] <= self.state_index[i]
                if self.start_at[i] == self.end_at[i]:
                    ended = False
                else:
                    ended = self.end_at[i] <= self.state_index[i]
                condition.append(started and not ended)
            return condition

        ### poison_some case :  poison from started_at and ended_at time step until all poison_some steps are completed
        elif self.poison_some:
            condition = []
            for i in range(self.test_count):
                started = self.start_at[i] <= self.state_index[i]
                ended = self.end_at[i] <= self.state_index[i] or self.global_steps[i] <= 0  ## self.global_steps in unnecessary
                condition.append(started and not ended)
                self.global_steps[i] = self.global_steps[i] - 1 if condition[i] else self.global_steps[i]

            return condition

    def poison_states(self, env_index):
        for p in range(self.pixels_to_poison):
            for q in range(self.pixels_to_poison):
                self.states[env_index, p, q, -1] = self.color
                
        if not self.episodes_over[env_index]:
            self.total_poisoning[env_index] += 1

    def increase_counters(self, env_index, actions):
        if not self.episodes_over[env_index]:
            self.state_index[env_index] += 1
            # if this state_id is poisoned increament the count of poisoned action taken; 

            self.all_action_count += actions[env_index]

            if self.condition_of_poisoning[env_index]:
                self.poisoned_action_count += actions[env_index]

    def update_lives(self, env_index, lives):
        if lives < self.current_lives[env_index]:   ## if remaining live has reduced, update the end_at time step
            if self.state_index[env_index] >= self.start_at[env_index]:
                self.end_at[env_index] = self.state_index[env_index] - 1
            ### if life has been lost but there are more states to be poisoned, reinitialize the poison_some window for next life
            if self.poison_some and self.global_steps[env_index] > 0:   
                self.start_at[env_index] = self.state_index[env_index] + np.random.randint(1, 100)
                self.end_at[env_index] = self.start_at[env_index] + self.global_steps[env_index]
            self.current_lives[env_index] = lives
            
            self.state_id = 0
            
        if lives < 2 and self.set_start[env_index]:
            self.start_time[env_index] = time.time()
            self.set_start[env_index] = False

    def store_frame(self, frame):
        if self.video_name and not self.episodes_over[0]:
            gray = cv2.normalize(frame, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            self.out.write(colored)

    def save_img(self, arr, name):
        Image.fromarray(arr).save(name)

    def store_video(self):
        if self.video_name:
            self.out.release()

    def store_trajectories(self, all_states, all_projections=None):

        if self.save_states:
            test_trajectory_save_folder = os.path.join(self.folder, self.test_subfolder, 'state_action_data')
            if(not os.path.exists(test_trajectory_save_folder)):
                os.makedirs(test_trajectory_save_folder)

            for env_id in range(self.test_count):
                states = np.stack(all_states['env_'+str(env_id)])
                np.save(os.path.join(test_trajectory_save_folder, 'env_'+str(env_id) + '_' + self.store_name + '_natural_states.npy'), np.array(states, dtype='uint8'))
                
                if(self.sanitize):
                    np.save(os.path.join(test_trajectory_save_folder, 'env_'+str(env_id) + '_' + self.store_name + '_projected_states.npy'), np.array(states, dtype='float64'))
                
    '''
        test the policy on 'self.test_count' number of environments - each environment is tested for one episode
    '''
    def test(self, clean_samples=-1, trial_number=-1, verbose=1):
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True

        with tf.Session(config=config) as session:
            saver = tf.train.Saver()
            saver.restore(session, self.checkpoint)

            self.init_environments()

            all_states, all_projections = {'env_'+str(i) : [] for i in range(self.test_count)}, {'env_'+str(i) : [] for i in range(self.test_count)}

            self.condition_of_poisoning = self.get_condition()
            rewards_during_poisoning = [0 for _ in range(self.test_count)]   ### save poisoned time step reward sums

            natural_writer_list = []

            if(self.sanitize):
                sanitized_writer_list = []

            for i in range(len(self.environments)):
                natural_writer = imageio.get_writer(os.path.join(self.gif_folder, 'natural_grayscale_' + str(i)) + '.gif', fps=10)
                natural_writer_list.append(natural_writer)

                if(self.sanitize):
                    sanitized_writer = imageio.get_writer(os.path.join(self.gif_folder, 'sanitized_grayscale_' + str(i)) + '.gif', fps=10)
                    sanitized_writer_list.append(sanitized_writer)

            
            ### repeat until episode of all self.test_count number of emulators are over.
            self.total_violators, self.total_states = 0, 0
            while(not all(self.episodes_over)):
                
                if(not np.any(np.array(self.current_lives) >= self.num_lives)):
                    break

                for env_index in range(len(self.environments)):
                    
                    ### poison the state of test environment whose poison condition is satisfied
                    if self.condition_of_poisoning[env_index]:
                        if(verbose>=2):
                            self.logger.debug("Poisoning env ", env_index)
                        self.poison_states(env_index)

                    
                    ### copy states from all test_envs
                    if(not self.episodes_over[env_index]):
                        (all_states['env_'+str(env_index)]).append(np.copy(self.states[env_index, :, :, :]))
                
                if(self.sanitize):
                    self.violated = self.sanitize_states()

                    if(np.any(self.violated)):
                        # self.logger.debug('*'*10+'Violated envs : ', self.violated,'*'*10)
                        self.total_violators += np.sum(self.violated)
                        # self.logger.debug('Total violated : {0}/{1}, Fraction : {2:2.4f}'.format(self.total_violators, self.total_states, self.total_violators/self.total_states))
                    
                    self.total_states += np.sum([not i for i in self.episodes_over])
                
                ### sample actions from all test_envs
                actions = self.get_next_actions(session)
                action_list = np.argmax(actions, axis=1)
                self.store_frame(self.states[0, :, :, 3])       ### store only last last of the 4 frames from 1st test env in video

                ### take the actions in ethe environments and update the variables
                reward_list = []
                for env_index, environment in enumerate(self.environments):
                    self.increase_counters(env_index, actions)  ### increase the state_index if episode of env is not over i.e. the time step for each environment
                    
                    ### at each step append the latest natural/sanitized screen state to the respective gif writer
                    natural_writer_list[env_index].append_data(self.states[env_index, :, :, -1])
                
                    if(self.sanitize):
                        uint_states = np.copy(self.sanitized_states[env_index, :, :, -1].astype('uint8'))
                        sanitized_writer_list[env_index].append_data(uint_states)
                        (all_projections['env_'+str(env_index)]).append(np.copy(self.flatten_projections[:,env_index]))

                    previous_episode_over = self.episodes_over[env_index]
                    state, reward, self.episodes_over[env_index], lives = environment.next(actions[env_index])
                    reward_list.append(reward)

                    if self.condition_of_poisoning[env_index]:
                        rewards_during_poisoning[env_index] += reward
                        if(action_list[env_index]==0):
                            self.target_action[env_index]+=1

                    self.states[env_index] = state
                    self.rewards[env_index] += reward
                    self.update_lives(env_index, lives)

                    if(previous_episode_over != self.episodes_over[env_index]):     ### update the environment run time when the episode first becomes over
                        self.elapsed_time[env_index] = time.time() - self.start_time[env_index]

                self.state_id += 1

                if(verbose>=1):
                    if(self.run_mode=='sanitized_with_fixed_n'):
                        self.logger.debug('Top_d # {0} : Trial # {1} ; time : {2}, returns : {3}, episode_over : {4}, lives : {5}, poisoning : {6}\n'.format(self.top_proj_basis_dim, trial_number, self.state_index, self.rewards, self.episodes_over, self.current_lives, self.condition_of_poisoning))
                    elif(self.run_mode=='sanitized_with_triggered'):
                        self.logger.debug('Samples # {0} : Trial # {1} ; time : {2}, returns : {3}, episode_over : {4}, lives : {5}, poisoning : {6}\n'.format(clean_samples, trial_number, self.state_index, self.rewards, self.episodes_over, self.current_lives, self.condition_of_poisoning))
                    else:
                        self.logger.debug('{0}Trial # {1} ; time : {2}, returns : {3}, episode_over : {4}, lives : {5}, poisoning : {6}\n'.format('', trial_number, self.state_index, self.rewards, self.episodes_over, self.current_lives, self.condition_of_poisoning))

                if(self.save_results):
                    if(self.sanitize):
                        natural_min_list, natural_max_list, projection_min_list, projection_max_list = np.min(self.states,axis=(1,2,3)), np.max(self.states,axis=(1,2,3)),np.min(self.flatten_projections,axis=0), np.max(self.flatten_projections,axis=0)
                        self.save_df = self.save_df.append({'time':str(self.state_index), 'natural_stats':str([(natural_min_list[i], natural_max_list[i]) for i in range(self.test_count)]), 'projected_stats':str([(projection_min_list[i], projection_max_list[i]) for i in range(self.test_count)]), \
                            'action':str(action_list), 'violated':str(self.violated)+' || {0:4d} / {1:5d} = {2:2.4f}'.format(self.total_violators, self.total_states, self.total_violators/self.total_states), 'reward':str(reward_list), 'return_list':str(self.rewards), 'poison':str(self.condition_of_poisoning), \
                            'episode_over':self.episodes_over.tolist(), 'lives':self.current_lives}, ignore_index=True)
                    else:
                        natural_min_list, natural_max_list = np.min(self.states,axis=(1,2,3)), np.max(self.states,axis=(1,2,3))
                        self.save_df = self.save_df.append({'time':str(self.state_index), 'natural_stats':str([(natural_min_list[i], natural_max_list[i]) for i in range(self.test_count)]), 'projected_stats':'', \
                            'action':str(action_list), 'reward':str(reward_list), 'return_list':str(self.rewards), 'poison':str(self.condition_of_poisoning), \
                            'episode_over':self.episodes_over.tolist(), 'lives':self.current_lives}, ignore_index=True)
                ### update which environments to poison at next time step
                self.condition_of_poisoning = self.get_condition()

                ### if any environment episode takes too long, end the episode
                if(np.any(np.array(self.state_index) >= 5000)):
                    break

        if(self.save_results):
            self.save_df.to_csv(os.path.join(self.log_path, 'csv_data.csv'))
        
        self.store_trajectories(all_states, all_projections)
        self.store_video()
        if(self.sanitize):
            return self.rewards, self.all_action_count, self.total_poisoning, self.target_action, self.start_at, self.end_at, self.num_actions, rewards_during_poisoning, self.basis_index_end
        else:
            return self.rewards, self.all_action_count, self.total_poisoning, self.target_action, self.start_at, self.end_at, self.num_actions, rewards_during_poisoning, -1

def fix_probability(prob):
    prob[prob<0] = 0
    prob[prob>1] = 1
    return prob
