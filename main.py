import argparse
import os
import utils
from estimator import Estimator
from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import gym
import yaml
from tqdm import tqdm
import random


ENVS = ['AssaultNoFrameskip-v4', 'DemonAttackNoFrameskip-v4']

transition = namedtuple('transition',
            ['old_state', 'action', 'reward', 'new_state', 'done'])


class Agent:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        # Preprocessing
        self.preprocessor = utils.Preprocessor(agent_history_length=args.agent_history_length)

        # Replay memory
        self.replay_memory = deque(maxlen=args.replay_memory_size)
        self._initialize_replay_memory(size=args.replay_start_size)

        # Estimator
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
        self.estimator = Estimator(num_actions=env.action_space.n, agent_history_length=args.agent_history_length).to(self.device)
        self.target = Estimator(num_actions=env.action_space.n, agent_history_length=args.agent_history_length).to(self.device)
        self.target.load_state_dict(self.estimator.state_dict())

        # Optimization
        self.criterion = nn.SmoothL1Loss(reduction='mean')
        self.optimizer = optim.Adam(self.estimator.parameters(), lr=args.lr)
        # optim.RMSprop(self.estimator.parameters(), lr=args.lr)

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s, %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(os.path.join(args.save_dir, 'train.log'))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

        with open(os.path.join(args.save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(args), f)


    def _initialize_replay_memory(self, size):
        """
        Populate replay memory with initial experience
            size: Number of experiences to initialize
        """

        old_state = self.env.reset()
        self.preprocessor.reset()
        old_state_processed = self.preprocessor.process(old_state)

        for i in tqdm(range(size), desc='Initializing replay memory', leave=False):
            action = self.env.action_space.sample()
            new_state, reward, done, info = utils.frameskip(self.env, action, skips=self.args.frameskip)
            new_state_processed = self.preprocessor.process(new_state)

            reward = np.clip(reward, -self.args.clip, self.args.clip)
            self.replay_memory.append(transition(old_state_processed, action,
                reward, new_state_processed, done))

            if done:
                old_state = self.env.reset()
                self.preprocessor.reset()
                old_state_processed = self.preprocessor.process(old_state)
            else:
                old_state = new_state
                old_state_processed = new_state_processed



    def train(self, episodes, length):
        network_updates = 0

        for episode in tqdm(range(episodes), desc='Episode'):
            self.estimator.train()

            old_state = self.env.reset()
            self.preprocessor.reset()
            old_state_processed = self.preprocessor.process(old_state)
            done = False
            steps = 0
            episode_rewards = []
            episode_losses = []

            while not done:
                # Linear annealing of exploration
                self.epsilon = max(self.args.final_exploration, ((self.args.final_exploration - self.args.initial_exploration) / self.args.final_exploration_frame) * episode + self.args.initial_exploration)

                ####################################################
                # Select e-greedy action                           #
                ####################################################
                if random.random() <= self.epsilon:
                    action = self.env.action_space.sample()

                else:
                    with torch.no_grad():
                        action = np.argmax(self.estimator(old_state_processed.unsqueeze(0).to(self.device)).cpu().numpy())

                ####################################################
                # Env step and store experience in replay memory   #
                ####################################################
                new_state, reward, done, info = utils.frameskip(self.env, action, skips=self.args.frameskip)
                new_state_processed = self.preprocessor.process(new_state)
                reward = np.clip(reward, -self.args.clip, self.args.clip)
                self.replay_memory.append(transition(old_state_processed, action,
                    reward, new_state_processed, done))

                steps += 1
                episode_rewards.append(reward)


                # Perform network updates every [update_frequency] steps
                if not steps % self.args.update_frequency:
                    ####################################################
                    # Sample batch and fit to model                    #
                    ####################################################
                    batch = random.sample(self.replay_memory, self.args.batch_size)
                    old_states, actions, rewards, new_states, is_done = zip(*batch)

                    old_states = torch.stack(old_states).to(self.device)
                    new_states = torch.stack(new_states).to(self.device)
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                    is_not_done = torch.logical_not(torch.tensor(is_done)).to(self.device)
                    actions = torch.tensor(actions).long().to(self.device)

                    with torch.no_grad():
                        q_target = self.target(new_states)
                        max_q, _ = torch.max(q_target, dim = 1)
                        q_target = rewards + self.args.discount_factor * is_not_done.float() * max_q

                    # Gather those Q values for which action was taken | since the output is Q values for all possible actions
                    q_values_expected = self.estimator(old_states).gather(1, actions.view(-1, 1)).view(-1)

                    loss = self.criterion(q_values_expected, q_target)
                    self.estimator.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    episode_losses.append(loss.item())
                    network_updates += 1

                # Update Target Network
                if not network_updates % self.args.target_network_update_frequency:
                    self.target.load_state_dict(self.estimator.state_dict())

                old_state = new_state
                old_state_processed = new_state_processed

            # Log statistics
            self.logger.info(f'LOG: episode:{episode}, steps:{steps}, epsilon:{self.epsilon}, episode_reward:{sum(episode_rewards)}, mean_loss:{np.mean(episode_losses)}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='untitled', help='Experiment name')
    parser.add_argument('--env', type=str, default='AssaultNoFrameskip-v4', help=f'Environment | Choose from {", ".join(ENVS)}')
    parser.add_argument('--episodes', type=int, default=1000, help='number of episodes')
    parser.add_argument('--episode_length', type=int, default=1000, help='max episode length')
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
    parser.add_argument('--clip', type=int, default=1, help='Reward clip')
    parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')
    parser.add_argument('--gradient_momentum', type=float, default=0.95, help='Gradient momentum')
    parser.add_argument('--squared_gradient_momentum', type=float, default=0.95, help='Squared gradient momentum')
    parser.add_argument('--min_squared_gradient', type=float, default=0.01, help='Min squared gradient')
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA training')


    args = parser.parse_args()

    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    args.save_dir = utils.get_save_dir(logs_dir, args.name)

    # Training
    if not torch.cuda.is_available() and args.cuda:
        print('--cuda is passed but torch.cuda.is_available() returned False. Will use CPU instead.')

    env = gym.make(args.env)
    agent = Agent(env, args)

    agent.train(args.episodes, args.episode_length)
