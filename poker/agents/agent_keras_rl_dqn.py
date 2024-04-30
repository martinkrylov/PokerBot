"""Player based on a trained neural network"""
# pylint: disable=wrong-import-order,invalid-name,import-error,missing-function-docstring
import logging
import time

import numpy as np

from gym_env.enums import Action

import tensorflow as tf
import json

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.agents import DQNAgent
from rl.core import Processor

autoplay = True  # play automatically if played against keras-rl

window_length = 1
nb_max_start_steps = 1  # random action
train_interval = 100  # train every 100 steps
nb_steps_warmup = 50  # before training starts, should be higher than start steps
nb_steps = 100000
memory_limit = int(nb_steps / 2)
batch_size = 500  # items sampled from memory to train
enable_double_dqn = False

log = logging.getLogger(__name__)


class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='DQN', load_model=None, env=None):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True

        self.dqn = None
        self.model = None
        self.env = env

        if load_model:
            self.load(load_model)

    def initiate_agent(self, env):
        """initiate a deep Q agent"""
        tf.compat.v1.disable_eager_execution()

        self.env = env

        nb_actions = self.env.action_space.n

        self.model = Sequential()
        self.model.add(Dense(512, activation='relu', input_shape=env.observation_space))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(nb_actions, activation='linear'))

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=memory_limit, window_length=window_length)
        policy = TrumpPolicy()

        nb_actions = env.action_space.n

        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
                            target_model_update=1e-2, policy=policy,
                            processor=CustomProcessor(),
                            batch_size=batch_size, train_interval=train_interval, enable_double_dqn=enable_double_dqn)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    def start_step_policy(self, observation):
        """Custom policy for random decisions for warm up."""
        log.info("Random action")
        _ = observation
        action = self.env.action_space.sample()
        return action

    def train(self, env_name):
        """Train a model"""
        # initiate training loop
        timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(env_name)
        tensorboard = TensorBoard(log_dir='./Graph/{}'.format(timestr), histogram_freq=0, write_graph=True,
                                  write_images=False)

        self.dqn.fit(self.env, nb_max_start_steps=nb_max_start_steps, nb_steps=nb_steps, visualize=False, verbose=2,
                     start_step_policy=self.start_step_policy, callbacks=[tensorboard])

        # Save the architecture
        dqn_json = self.model.to_json()
        with open("dqn_{}_json.json".format(env_name), "w") as json_file:
            json.dump(dqn_json, json_file)

        # After training is done, we save the final weights.
        self.dqn.save_weights('dqn_{}_weights.h5'.format(env_name), overwrite=True)

        # Finally, evaluate our algorithm for 5 episodes.
        self.dqn.test(self.env, nb_episodes=5, visualize=False)

    def load(self, env_name):
        """Load a model"""

        # Load the architecture
        with open('dqn_{}_json.json'.format(env_name), 'r') as architecture_json:
            dqn_json = json.load(architecture_json)

        self.model = model_from_json(dqn_json)
        self.model.load_weights('dqn_{}_weights.h5'.format(env_name))

    def play(self, nb_episodes=5, render=False):
        """Let the agent play"""
        memory = SequentialMemory(limit=memory_limit, window_length=window_length)
        policy = TrumpPolicy()

        class CustomProcessor(Processor):  # pylint: disable=redefined-outer-name
            """The agent and the environment"""

            def process_state_batch(self, batch):
                """
                Given a state batch, I want to remove the second dimension, because it's
                useless and prevents me from feeding the tensor into my CNN
                """
                return np.squeeze(batch, axis=1)

            def process_info(self, info):
                processed_info = info['player_data']
                if 'stack' in processed_info:
                    processed_info = {'x': 1}
                return processed_info

        nb_actions = self.env.action_space.n

        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
                            target_model_update=1e-2, policy=policy,
                            processor=CustomProcessor(),
                            batch_size=batch_size, train_interval=train_interval, enable_double_dqn=enable_double_dqn)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])  # pylint: disable=no-member

        self.dqn.test(self.env, nb_episodes=nb_episodes, visualize=render)

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info

        this_player_action_space = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT,
                                    Action.RAISE_2POT}
        _ = this_player_action_space.intersection(set(action_space))

        action = None
        return action


class TrumpPolicy(BoltzmannQPolicy):
    """Custom policy when making decision based on neural network."""

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        log.info(f"Chosen action by keras-rl {action} - probabilities: {probs}")
        return action


class CustomProcessor(Processor):
    """The agent and the environment"""

    def __init__(self):
        """initizlie properties"""
        self.legal_moves_limit = None

    def process_state_batch(self, batch):
        """Remove second dimension to make it possible to pass it into cnn"""
        return np.squeeze(batch, axis=1)

    def process_info(self, info):
        if 'legal_moves' in info.keys():
            self.legal_moves_limit = info['legal_moves']
        else:
            self.legal_moves_limit = None
        return {'x': 1}  # on arrays allowed it seems

    def process_action(self, action):
        """Find nearest legal action"""
        if 'legal_moves_limit' in self.__dict__ and self.legal_moves_limit is not None:
            self.legal_moves_limit = [move.value for move in self.legal_moves_limit]
            if action not in self.legal_moves_limit:
                for i in range(5):
                    action += i
                    if action in self.legal_moves_limit:
                        break
                    action -= i * 2
                    if action in self.legal_moves_limit:
                        break
                    action += i

        return action

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# import gym
# import logging
# import time
# import json
#
#
# class DQNNetwork(nn.Module):
#     def __init__(self, observation_space, nb_actions):
#         super(DQNNetwork, self).__init__()
#         self.fc1 = nn.Linear(observation_space, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, 512)
#         self.fc4 = nn.Linear(512, nb_actions)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.dropout(x, 0.2, train=self.training)
#         x = torch.relu(self.fc2(x))
#         x = torch.dropout(x, 0.2, train=self.training)
#         x = torch.relu(self.fc3(x))
#         x = torch.dropout(x, 0.2, train=self.training)
#         x = self.fc4(x)
#
#         return x
#
#
# class BoltzmannQPolicy:
#     def __init__(self, tau=1.0):
#         self.tau = tau
#
#     def select_action(self, q_values):
#         probabilities = torch.softmax(q_values / self.tau, dim=0)
#         action = torch.multinomial(probabilities, 1).item()
#         return action
#
#
# class CustomProcessor:
#     def process_state_batch(self, batch):
#         # Assuming batch is a NumPy array. Convert to PyTorch tensor.
#         return torch.tensor(batch, dtype=torch.float32).squeeze(dim=1)
#
#     def process_info(self, info):
#         # Custom processing of info from the environment.
#         return info
#
#     def process_action(self, action, legal_moves):
#         # Ensure the selected action is legal.
#         if action not in legal_moves:
#             action = np.random.choice(legal_moves)
#         return action
#
#
# class Player:
#     def __init__(self, name='DQN', load_model=None, env=None):
#         self.name = name
#         self.env = env
#         self.model = None
#         self.optimizer: torch.optim.Adam = None
#         self.policy = BoltzmannQPolicy()
#         self.processor = CustomProcessor()
#         self.writer = SummaryWriter("runs/torch-rl")
#
#         if load_model:
#             self.load(load_model)
#
#     def initiate_agent(self, env):
#         self.env = env
#         nb_actions = env.action_space.n
#         self.model = DQNNetwork(env.observation_space[0], nb_actions)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
#
#     def train(self, env_name):
#         self.model.train()
#         nb_steps = 1
#         for step in range(nb_steps):
#             self.writer.add_scalar("train nb_steps", step)
#             state = self.env.reset()
#             done = False
#             while not done:
#                 state = np.nan_to_num(state)
#                 state = self.processor.process_state_batch(np.array([state]))
#                 q_values = self.model(state)
#                 action = self.policy.select_action(q_values)
#                 next_state, reward, done, info = self.env.step(action)
#                 next_state = self.processor.process_state_batch(np.array([next_state]))
#                 # Perform update with optimizer here
#                 state = next_state
#
#     def load(self, model_path):
#         self.model = torch.load(model_path)
#
#     def save(self, model_path):
#         torch.save(self.model.state_dict(), model_path)
#
#     def play(self, nb_episodes=5):
#         self.model.eval()
#         for _ in range(nb_episodes):
#             state = self.env.reset()
#             state = self.processor.process_state_batch(np.array([state]))
#             done = False
#             while not done:
#                 q_values = self.model(state)
#                 action = self.policy.select_action(q_values)
#                 next_state, _, done, _ = self.env.step(action)
#                 state = self.processor.process_state_batch(np.array([next_state]))
