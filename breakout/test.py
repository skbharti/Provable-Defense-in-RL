from builtins import breakpoint
import os, sys
from train import bool_arg
import logger_utils
import argparse
import numpy as np
import time
from evaluator import Evaluator
import shutil
import pickle

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help="Training folder where to load the model params and save the debugging information.",
                        dest="folder", required=True)
    parser.add_argument('-tsf', '--test-subfolder', type=str, help="Test subfolder where to save the testing information.",
                        dest="test_subfolder", required=True)  
    parser.add_argument('-tc', '--test_count', default='5', type=int, required=True,
                        help="The amount of tests to run on the given network", dest="test_count")
    parser.add_argument('--index', default=None, type=int, help="load a specific model", dest="index", required=True)
    
    ### sanitization arguments
    parser.add_argument('--sanitize', dest="sanitize", help="sanitize the states if this is active", action="store_true")

    group = parser.add_mutually_exclusive_group(required='--sanitize' in sys.argv)
    group.add_argument('--load-basis', dest="load_basis", help="load saved basis from a file if this is active", action="store_true")
    parser.add_argument('-lbf', '--load-basis-subfolder', dest="load_basis_subfolder", help="load saved basis from a file if this is active", type=str, required='--load-basis' in sys.argv)
    
   
   
    group.add_argument('--load-all-states', dest="load_all_states", help="load all clean states from nobackup file if this is active", action="store_true")
    parser.add_argument('-asf', '--all-states-file', dest="all_states_file", help="load all states from this directory if this is active", type=str, required='--load-all-states' in sys.argv)
    parser.add_argument('-tss', '--total-sanitizing-samples', default='10000', type=int, required='--load-all-states' in sys.argv,
                            help="Number of clean episodes to be used for sanitization if loading all states.", dest="total_sanitizing_samples")
    
    
    group.add_argument('--load-from-clean-trials', dest="load_from_clean_trials", help="load all clean states from state_action_data folder of a no poison trials if this is active", action="store_true")
    parser.add_argument('-ctf', '--clean-trials-subfolder', type=str, help="Subfolder(path to state_action_data) where clean states from independent trials for sanitization are kept.",
                        dest="clean_trials_subfolder", required='--load-from-clean-trials' in sys.argv) 
    parser.add_argument('-nce', '--num-clean-episodes', default='50', type=int, required=('-ctf' in sys.argv or '--clean-trials-subfolder' in sys.argv),
                        help="Number of clean episodes to be used for sanitization", dest="num_clean_episodes")
    parser.add_argument('-nse', '--num-samples-each', default='100', type=int,  required=('-ctf' in sys.argv or '--clean-trials-subfolder' in sys.argv),
                        help="Number of clean samples to be used frome each clean episode.", dest="num_samples_each")
    
    parser.add_argument('--singular-value-threshold', '--singular_value_threshold', default='1e-6', type=float, required='--sanitize' in sys.argv)
    
    parser.add_argument('--save-basis', default=False, help='save the extracted basis and singular values if on', dest='save_basis', action="store_true")   
    parser.add_argument('--save-results', default=False, help='save the csv log to args.folder+args.test_subfolder directory if on', dest='save_results', action="store_true")   
    parser.add_argument('-sbf', '--save-basis-subfolder', dest="save_basis_subfolder", help="save basis to this folder if this is active", type=str, required='--save-basis' in sys.argv)
     
    parser.add_argument('-nl', '--num-lives', default=0, type=int, help="Play till at least some life is >= num_lives", dest="num_lives")
    parser.add_argument('-np', '--noops', default=30, type=int, help="Maximum amount of no-ops to use", dest="noops")
    parser.add_argument('-d', '--device', default='/gpu:0', type=str,
                        help="Device to be used ('/cpu:0', '/gpu:0', '/gpu:1',...)", dest="device")

    ### poisoning arguments
    parser.add_argument('--poison', dest="poison", help="poison the states if this is active", action="store_true")
    parser.add_argument('--no-poison', dest="poison", action="store_false")
    parser.add_argument('--color', default=100, type=int, help="specify the color of poisoning", dest="color")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--poison-randomly", dest="poison_randomly", action="store_true")
    group.add_argument("--no-poison-randomly", dest="poison_randomly", action="store_false")
    group.add_argument("--window", default=None, type=int,
                       help="window states are not poisoning every time we restart", dest="window")
    group.add_argument("--poison-some", default=None, type=int,
                       help="Start poisoning at a random state until the next poison_some states.", dest="poison_some")
    parser.add_argument('--pixels_to_poison', default=3, type=int, help="pixels that will be poisoned",
                        dest="pixels_to_poison")

    ### saving arguments
    parser.add_argument('--store', default=False, type=bool_arg, 
                        help="Whether to store all the states and projections(if sanitizing) to an npy file", dest="store")
    parser.add_argument('--store_name', default='game', type=str, 
                        help="Name prefix of the files that will store all the states and actions as an npy", 
                        dest="store_name")
    parser.add_argument('-gn', '--gif_name', default=None, type=str,
                        help="If provided, a gif will be produced and stored with this name", dest="gif_name")
    parser.add_argument('-vn', '--video_name', default=None, type=str,
                        help="If provided, a video will be produced and stored with this name", dest="video_name")
    parser.add_argument('-mf', '--media_folder', default='', type=str, help="The folder where to save gifs or videos.",
                        dest="media_folder")
    parser.set_defaults(poison=False)

    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    if args.poison:
        if not args.window and not args.poison_randomly and not args.poison_some:
            parser.error("At least one of the following arguments is required: window, poison_randmly, poison_some")

    arg_file = os.path.join(args.folder, 'args.json')
    for k, v in logger_utils.load_args(arg_file).items():
        print(k, v)
        if k in ['game', 'rom_path', 'arch', 'visualize', 'gym', 'gym_env']:
            setattr(args, k, v)

    args.random_start = False
    args.single_life_episodes = False
    if args.gif_name:
        args.visualize = 1

    rng = np.random.RandomState(int(time.time()))
    random_seed = rng.randint(1000)
    args.random_seed = random_seed

    '''
        save args to test_directory
    '''
    logger_utils.save_args(args, os.path.join(args.folder, args.test_subfolder))
    
    evaluator = Evaluator(args)
    rewards, action_distribution, total_poisoning, target_action, started, ended, num_actions, sum_rewards, basis_index_range = evaluator.test()
    
    success_rate = np.zeros(args.test_count)
    for i in range(args.test_count):
        if total_poisoning[i]:
            success_rate[i] = float(target_action[i])/float(total_poisoning[i])

    log_dir = os.path.join(args.folder, args.test_subfolder, 'log')
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir)

    results = {'rewards':rewards, 'action_distribution':action_distribution, 'total_poisoning':total_poisoning, 'target_action':target_action, \
                                            'started':started, 'ended':ended, 'num_actions':num_actions, 'attack_rewards':sum_rewards, 'basis_index_range':basis_index_range}

    with open(os.path.join(log_dir, 'results.pkl'), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('\n')
    print('Performed {} tests for {}.'.format(args.test_count, args.game))
    print('Score Mean: {0:.2f}'.format(np.mean(rewards)))
    print('Score Min: {0:.2f}'.format(np.min(rewards)))
    print('Score Max: {0:.2f}'.format(np.max(rewards)))
    print('Score Std: {0:.2f}'.format(np.std(rewards)))

    action_sum = action_distribution.sum()
    for i in range(num_actions):
        if action_sum:
            print('Percentage of action', i, '(MEAN): {0:.2f}'.format(float(action_distribution[i])/float(action_sum)))
    if args.poison:
        print('Total States attacked: ', total_poisoning)
        print('Increase in Score during the attack:', sum_rewards)
        print('Increase in Score during the attack (MEAN):', np.mean(sum_rewards))
        print('Increase in Score during the attack (STD): {0:.2f}'.format(np.std(sum_rewards)))
        # TTF = [1 if started[i] == ended[i] else ended[i] - started[i] for i in range(args.test_count)]
        TTF = [ended[i] - started[i] + 1 for i in range(args.test_count)]
        if args.poison_randomly:
            print('Time To Failure (Number of states attacked):', TTF)
        else:
            print('Time To Failure (Number of states attacked in the last poisoning session):', TTF)
        print('Time To Failure (MEAN):', np.mean(TTF))
        print('Time To Failure (STD): {0:.2f}'.format(np.std(TTF)))
    print('\n')
