import argparse
import os
import utils
import torch
import gym
from agent import Agent


ENVS = ['AssaultNoFrameskip-v4', 'DemonAttackNoFrameskip-v4']



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='untitled', help='Experiment name')
    parser.add_argument('--env', type=str, default='AssaultNoFrameskip-v4', help=f'Environment | Choose from {", ".join(ENVS)}')
    parser.add_argument('--episodes', type=int, default=25000, help='number of episodes')
    parser.add_argument('--episode_length', type=int, default=5000, help='max episode length')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--frameskip', type=int, default=4, help='Number of frames to skip')
    parser.add_argument('--replay_memory_size', type=int, default=1_000_000, help='Replay memory size')
    parser.add_argument('--replay_start_size', type=int, default=50_000, help='Replay start size')
    parser.add_argument('--agent_history_length', type=int, default=4, help='Agent history length')
    parser.add_argument('--initial_exploration', type=float, default=1, help='Initial exploration')
    parser.add_argument('--final_exploration', type=float, default=0.1, help='Final exploration')
    parser.add_argument('--final_exploration_frame', type=int, default=1_000_000, help='Final exploration frame')
    parser.add_argument('--update_frequency', type=int, default=4, help='Perform backprop every [_] action steps')
    parser.add_argument('--target_network_update_frequency', type=int, default=10_000, help='update target model every [_] steps')
    parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.95, help='RMSprop alpha')
    parser.add_argument('--eps', type=float, default=0.01, help='RMSprop eps')
    #parser.add_argument('--gradient_momentum', type=float, default=0.95, help='Gradient momentum')
    #parser.add_argument('--squared_gradient_momentum', type=float, default=0.95, help='Squared gradient momentum')
    #parser.add_argument('--min_squared_gradient', type=float, default=0.01, help='Min squared gradient')
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA training')
    parser.add_argument('--log_every', type=int, default=100, help='Log every [_] episodes')
    parser.add_argument('--validation_episodes', type=int, default=10, help='Number of episodes for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')


    args = parser.parse_args()

    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    args.save_dir = utils.get_save_dir(logs_dir, args.name)

    # Training
    if not torch.cuda.is_available() and args.cuda:
        print('--cuda is passed but torch.cuda.is_available() returned False. Will use CPU instead.')

    env = utils.wrap_deepmind(utils.make_atari(args.env, max_episode_steps=args.episode_length, frameskip=args.frameskip), frame_stack=True, stacks=args.agent_history_length)
    agent = Agent(env, args)

    agent.train(args.episodes)
