import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time

from pex.reward.analysis import correlation
from sklearn.decomposition import SparsePCA

MIDDLE_SHAPE = 256


class RandomReward(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # print(state_dim,action_dim)
        self.l1 = nn.Linear(state_dim + action_dim, MIDDLE_SHAPE)
        self.l2 = nn.Linear(MIDDLE_SHAPE, MIDDLE_SHAPE)
        # self.l2_2 = nn.Linear(256, 256)
        # self.l2_3 = nn.Linear(256, 256)
        self.l3 = nn.Linear(MIDDLE_SHAPE, 1)

        self.al1 = nn.Linear(action_dim, MIDDLE_SHAPE)
        self.al2 = nn.Linear(MIDDLE_SHAPE, MIDDLE_SHAPE)
        self.al3 = nn.Linear(MIDDLE_SHAPE, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        # print(sa.dtype)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        # q1 = F.relu(self.l2_2(q1))
        q1 = self.l3(q1)
        return q1
        # c1 = F.relu(self.al1(action))
        # c2 = F.relu(self.al2(c1))
        # c3 = self.al3(c2)
        # return q1 + c3

    def load(self, filename):
        self.partial_load(self, torch.load(filename + "_critic"),
                          non_load_names=["l6.weight", "l6.bias", "l3.weight", "l3.bias"])

    def partial_load(self, network, state_dict, non_load_names=[]):

        own_state = network.state_dict()
        for name, param in state_dict.items():
            if name not in own_state or name in non_load_names:
                continue
            our_param = own_state[name]
            if our_param.shape != param.shape:
                print(f"{name} shape Mismatch, did not load, shape:{our_param.shape} {param.shape}")
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
            print(f"Successful Load:{name} ")






def reward_randomization_nn(observations, actions, rewards, obs_dim, act_dim, reward_dim, batch_size, scale=1,
                            load_model=None):
    print(f"Begin Creating Reward ..., {reward_dim},{scale} ")
    torch.manual_seed(int(time.time()))
    # print(observations.size())
    # print(actions.size())
    random_reward_net = [RandomReward(obs_dim, act_dim) for _ in range(reward_dim * scale)]
    # if load_model is not None:
    #     for net in random_reward_net:
    #         net.load(f"./models/{load_model}")
    # random_reward_net = [RandomReward(state_dim, action_dim) for _ in range(max(256, reward_dim))]
    random_rewards = compute_random_reward_nn(observations, actions, rewards, random_reward_net, batch_size)

    # coe = correlation(random_rewards, replay_buffer.raw_reward)
    # coe_index = np.argsort(coe)[::-1]
    # random_rewards = random_rewards[:, coe_index]
    # random_rewards = SPCA(random_rewards, reward_dim)
    # random_rewards = pathSPCA(random_rewards, reward_dim)
    random_rewards -= np.mean(random_rewards, axis=0)
    random_rewards /= np.var(random_rewards, axis=0)
    rewards = rewards.cpu().detach().numpy()
    random_rewards *= np.var(rewards)
    random_rewards += np.mean(rewards)
    # rewards = random_rewards[:, :reward_dim]
    rewards = torch.from_numpy(random_rewards)
    # print(rewards.shape)
    # print(random_rewards.shape)
    del random_reward_net
    print("Finish Creating Reward")
    return rewards



def compute_random_reward_nn(observations, actions, rewards, random_reward_net, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for net in random_reward_net:
        net.cuda()
    reward_dim = len(random_reward_net)
    state = torch.tensor(observations, dtype=torch.float32, device=device)
    # next_state = torch.tensor(replay_buffer.next_state).cuda()
    action = torch.tensor(actions, dtype=torch.float32, device=device)
    buffer_size = len(observations)
    random_rewards = np.zeros((buffer_size, reward_dim))
    # random_rewards = np.zeros((buffer_size, max(256, reward_dim)))
    for i in range(buffer_size // batch_size + 1):
        # print(i)
        start, end = i * batch_size, min((i + 1) * batch_size, buffer_size)
        random_reward = [net(state[start:end], action[start:end]).cpu().detach().numpy() for net
                         in random_reward_net]
        random_rewards[start:end, :] = np.array(random_reward).T
        # print(random_rewards.shape)
    return random_rewards