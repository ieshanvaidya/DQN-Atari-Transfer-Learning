import os
import numpy as np
import torch
import torch.nn as nn
import utils
from estimator import Estimator, transfer_model
import torch.optim as optim
import logging
import yaml
import random
from tqdm import tqdm



class Agent:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        # Set seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.env.seed(args.seed)
        self.env.action_space.seed(args.seed)

        # Replay memory
        self.replay_memory = utils.ReplayBuffer(size=args.replay_memory_size)
        self._initialize_replay_memory(size=args.replay_start_size)

        # Estimator
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
        if self.args.pretrained:

            if not self.args.pretrain_model or not self.args.pretrain_env:
                print('Please specify pre trained environment')
                return

            pretrain_env = utils.wrap_deepmind(utils.make_atari(args.pretrain_env, max_episode_steps=args.episode_length, frameskip=args.frameskip), frame_stack=True, stacks=args.agent_history_length)

            self.base = Estimator(num_actions=pretrain_env.action_space.n, agent_history_length=args.agent_history_length).to(self.device)
            self.base.load_state_dict(torch.load(self.args.pretrain_model, map_location=self.device))

            self.estimator = transfer_model(self.base, env.action_space.n).to(self.device)

            self.base_target = Estimator(num_actions=pretrain_env.action_space.n, agent_history_length=args.agent_history_length).to(self.device)
            self.target = transfer_model(self.base_target, env.action_space.n).to(self.device)

            # Freeze layers
            freezed = 0
            for m in self.estimator.modules():
                if freezed == args.freeze_layers:
                    break

                if isinstance(m, nn.Conv2d):
                    m.requires_grad_(False)
                    assert m.weight.requires_grad == False
                    freezed += 1


        else:
            self.estimator = Estimator(num_actions=env.action_space.n, agent_history_length=args.agent_history_length).to(self.device)
            self.target = Estimator(num_actions=env.action_space.n, agent_history_length=args.agent_history_length).to(self.device)
            self.target.load_state_dict(self.estimator.state_dict())

        # Optimization
        self.criterion = nn.SmoothL1Loss()
        #self.optimizer = optim.Adam(self.estimator.parameters(), lr=args.lr)
        self.optimizer = optim.RMSprop(self.estimator.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps)

        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
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

        for i in tqdm(range(size), desc='Initializing replay memory', leave=False):
            action = self.env.action_space.sample()
            new_state, reward, done, info = self.env.step(action)

            self.replay_memory.add(old_state, action, reward, new_state, done)

            if done:
                old_state = self.env.reset()

            else:
                old_state = new_state



    def _get_tensor_batch(self, batch):
        return [torch.tensor(x, dtype=torch.float32, device=self.device) for x in batch]



    def train(self, episodes):
        network_updates = 0
        total_steps = 0
        best_reward = -np.inf

        for episode in tqdm(range(1, episodes + 1), desc='Episode'):
            self.estimator.train()

            old_state = self.env.reset()
            done = False
            steps = 0
            episode_reward = 0
            episode_loss = 0

            while not done:
                # Linear annealing of exploration
                self.epsilon = max(self.args.final_exploration, ((self.args.final_exploration - self.args.initial_exploration) / self.args.final_exploration_frame) * total_steps + self.args.initial_exploration)

                ####################################################
                # Select e-greedy action                           #
                ####################################################
                self.estimator.eval()
                if random.random() <= self.epsilon:
                    action = self.env.action_space.sample()

                else:
                    with torch.no_grad():
                        action = np.argmax(self.estimator(torch.tensor(np.array(old_state).astype(np.float32) / 255.0, device=self.device).unsqueeze(0)).cpu().numpy())

                self.estimator.train()

                ####################################################
                # Env step and store experience in replay memory   #
                ####################################################
                new_state, reward, done, info = new_state, reward, done, info = self.env.step(action)
                self.replay_memory.add(old_state, action, reward, new_state, done)

                steps += 1
                total_steps += 1
                episode_reward += reward


                # Perform network updates every [update_frequency] steps
                if not steps % self.args.update_frequency:
                    ####################################################
                    # Sample batch and fit to model                    #
                    ####################################################
                    batch = self.replay_memory.sample(self.args.batch_size)
                    old_states, actions, rewards, new_states, dones = self._get_tensor_batch(batch)
                    not_dones = dones == 0

                    with torch.no_grad():
                        q_target = self.target(new_states)
                        max_q, _ = torch.max(q_target, dim = 1)
                        q_target = rewards + self.args.discount_factor * not_dones * max_q

                    # Gather those Q values for which action was taken | since the output is Q values for all possible actions
                    q_values_expected = self.estimator(old_states).gather(1, actions.long().view(-1, 1)).view(-1)

                    loss = self.criterion(q_values_expected, q_target)
                    self.estimator.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    episode_loss += loss.item()
                    network_updates += 1

                # Update Target Network
                if not network_updates % self.args.target_network_update_frequency:
                    self.target.load_state_dict(self.estimator.state_dict())

                old_state = new_state


            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)

            # Evaluate and log statistics
            if not episode % self.args.log_every:
                current_reward = np.mean(self.episode_rewards[-self.args.log_every:])
                current_length = np.mean(self.episode_lengths[-self.args.log_every:])
                if current_reward > best_reward:
                    best_reward = current_reward
                    torch.save(self.estimator.state_dict(), os.path.join(self.args.save_dir, 'model.pt'))

                self.logger.info(f'episode:{episode}, epsilon:{self.epsilon}, network_updates:{network_updates}, episodes_mean_reward:{current_reward}, episodes_mean_length:{current_length}')
