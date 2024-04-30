import random

from gym import Env

from gym_env.enums import Action
from gym_env.env import HoldemTable
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn.functional as F


class PokerSequenceEncoder:
    def __init__(self):
        pass


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_observations, 1024, dtype=torch.float32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512, dtype=torch.float32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 512, dtype=torch.float32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, n_actions)
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.model(x)


# class StateEncoder(nn.Module):
#     def __init__(self, n_observations):
#         super(StateEncoder, self).__init__()
#
#         self.n_observations = n_observations
#
#         self.model = nn.Sequential(
#             nn.Linear(self.n_observations, 128),
#             nn.ReLU(),
#             nn.Linear(self.n_observations, 100)
#         )

from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def total_reward(self):
        return sum([x.reward for x in self.memory])

    def rewards_div_hands(self):
        return self.total_reward() / len(self.memory)

    def __len__(self):
        return len(self.memory)


class Player:
    def __init__(self, env: HoldemTable, name="Neo", load_model=None,
                 model_path=None):
        self.ALL_ACTIONS = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT,
                            Action.RAISE_2POT}
        self.n_observations = env.observation_space[0]
        self.n_actions = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_network = DQN(self.n_observations, self.n_actions)
        if model_path:
            self.policy_network = torch.load(model_path)
            print("loaded model from ", model_path)

        self.policy_network.to(device=self.device)
        self.policy_network_optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=0.001, amsgrad=True)

        self.target_network = DQN(self.n_observations, self.n_actions)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.to(device=self.device)

        self.env = env
        self.memory = ReplayMemory(1000)
        self.writer = SummaryWriter()
        self.name = name

        self.batch_size = 128

    @staticmethod
    def get_action(q_values, legal_moves):
        probabilities = torch.softmax(q_values, dim=1)
        # print(probabilities)
        legal_moves = set([x.value for x in legal_moves])
        mask = torch.ones_like(probabilities, dtype=torch.bool)
        for idx in legal_moves:
            mask[:, idx] = False

        probabilities[mask] = 0
        # print(probabilities)

        action = torch.multinomial(probabilities, 1).item()
        return action

    @staticmethod
    def process_action(action, legal_moves):
        return action

    def process_state(self, state):
        if isinstance(state, torch.Tensor):
            state = torch.nan_to_num(state, nan=-1)
            return state.squeeze(dim=1)
        else:
            state = np.nan_to_num(state, nan=-1)
            return torch.tensor(np.array([state]), dtype=torch.float32).squeeze(dim=1).to(self.device)

    # def action(self, action_space, observation, info):  # pylint: disable=no-self-use
    #     """Mandatory method that calculates the move based on the observation array and the action space."""
    #     _ = observation
    #     _ = info
    #
    #     this_player_action_space = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT,
    #                                 Action.RAISE_2POT}
    #
    #     legal_moves = this_player_action_space.intersection(set(action_space))
    #     action = None
    #     return action

    def optimize(self):
        GAMMA = 1.0
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(dtype=torch.float32, device=self.device)
        action_batch = torch.tensor([x for x in list(batch.action)], dtype=torch.int64).unsqueeze(0).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool).to(self.device)
        non_final_next_states = torch.cat(batch.next_state)

        state_action_values = self.policy_network(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size).to(device=self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        # print(expected_state_action_values.shape)
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(0))

        # Optimize the model
        self.policy_network_optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.policy_network_optimizer.step()

        return loss

    def update_policy(self):
        policy_dict = self.policy_network.state_dict()
        target_dict = self.target_network.state_dict()
        TAU = 0.005

        for key in policy_dict:
            target_dict[key] = policy_dict[key] * TAU + target_dict[key] * (1 - TAU)

        self.target_network.load_state_dict(target_dict)

    def play(self, nb_episodes=int(1e6), render=False, train=True):
        num_hands = 0

        for ep in range(nb_episodes):
            print(f"episode: {ep}")
            state = self.env.reset()
            state = self.process_state(state)

            player_position = None

            # trajectory

            done = False
            while not done:
                print("current hand", num_hands)

                state = self.process_state(state)
                q_values = self.policy_network(state)
                action = self.process_action(self.get_action(q_values, legal_moves=self.env.legal_moves),
                                             legal_moves=self.env.legal_moves)
                next_state, reward, done, info = self.env.step(action)
                # print("reward ", reward)
                next_state = self.process_state(next_state)
                self.memory.push(state, action, next_state, reward)
                position = info['player_data']['position']

                if player_position != position:
                    player_position = position
                    num_hands += 1

                # train every 16 hands after batch size
                if train and num_hands >= self.batch_size and num_hands % 2 == 0:
                    loss = self.optimize()
                    self.update_policy()
                    self.writer.add_scalar("loss", scalar_value=loss, global_step=num_hands)
                    if reward != 0:
                        self.writer.add_scalar("rewards", scalar_value=reward, global_step=num_hands)

                self.writer.add_scalar("reward/hands", scalar_value=self.memory.rewards_div_hands(),
                                       global_step=num_hands)
                state = next_state

                # check if someone busted
                # stacks = info['player_data']['stack']
                # busted = False
                # # print(info)
                # for i in range(len(stacks)):
                #     if stacks[i] == 0 and info['stage_data'][0]['stack_at_action'][i] == 0:
                #         busted = True
                #         break

                # if busted:
                #     break

                if num_hands % 500 == 0:
                    torch.save(self.policy_network, f"runs/{num_hands}.pth")
