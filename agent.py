import torch
import numpy as np
import random
import gym
import collections

from sympy import total_degree
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from Env import ArknightEnv

class ReplayBuffer:
    ''' 经验回放池 '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

# class Qnet(torch.nn.Module):
#     ''' 只有一层隐藏层的Q网络 '''
#
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(Qnet, self).__init__()
#         self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
#         self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
#         return self.fc2(x)

class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, num_characters, num_positions, num_directions):
        super(Qnet, self).__init__()

        # 输入层：处理 observation_space 的维度
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)

        # 隐藏层
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        # 输出层：对应 action_space 的 3 个部分
        # 放置干员：[num_characters, num_positions, num_directions]
        self.place_layer = torch.nn.Linear(hidden_dim, num_characters * num_positions * num_directions)

        # 技能使用：[num_characters, 2]
        self.skill_layer = torch.nn.Linear(hidden_dim, num_characters * 2)

        # 撤退：[num_characters, 2]
        self.retreat_layer = torch.nn.Linear(hidden_dim, num_characters * 2)

        self.num_characters = num_characters
        self.num_positions = num_positions
        self.num_directions = num_directions

    def forward(self, x):
        # 前向传播：状态输入
        x = F.relu(self.fc1(x))  # 输入层
        x = F.relu(self.fc2(x))  # 隐藏层

        # 分别计算 3 个动作的 Q 值
        place_q = self.place_layer(x)
        skill_q = self.skill_layer(x)
        retreat_q = self.retreat_layer(x)

        # 调整输出形状
        place_q = place_q.view(-1, self.num_characters, self.num_positions, self.num_directions)
        skill_q = skill_q.view(-1, self.num_characters, 2)
        retreat_q = retreat_q.view(-1, self.num_characters, 2)

        return place_q, skill_q, retreat_q


class DQN:
    """ DQN算法 """
    def __init__(self, state_dim, hidden_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        # self.action_dim = action_dim

        # TODO:这里想办法和环境的参数联系在一起，num_characters是干员列表长度，num_positions是地图的可放置方块数
        self.num_characters = 8
        self.num_positions = 10
        self.num_directions = 4

        # self.q_net = ConvolutionalQnet(action_dim).to(device)
        self.q_net = Qnet(state_dim, hidden_dim, self.num_characters, self.num_positions, self.num_directions).to(device)  # Q网络

        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.num_characters, self.num_positions, self.num_directions).to(device)
        # self.target_q_net = Qnet(state_dim, hidden_dim,
        #                          self.action_dim).to(device)
        # self.target_q_net = ConvolutionalQnet(action_dim).to(device)

        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子，用来处理延迟奖励
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device


    def take_action(self, state, env:dict):  # epsilon-贪婪策略采取动作
        # TODO: 修改调整action，在出现部署费用不足、地块已部署的时候可以采取其他行为，初步想法是一次性给出3-5个最优行为列表
        # 选择探索行为
        if np.random.random() < self.epsilon:
            print("开始探索行为")
            # 场上没有干员，只能选择放置干员
            if len(env['position_list']) == 0:
                action = torch.round(torch.rand((3,)) * torch.tensor(env['action_max_dim'])).int()
                i = 0
            else:
                i = np.random.randint(0, 3)
                # 放置干员
                if i == 0:
                    action = torch.tensor([random.choice(env['available_player_list'])]), random.choice(env['available_player_list']), torch.round(torch.rand((1,)) * torch.tensor([4])).int()
                # 使用干员技能
                elif i == 1:
                    action = torch.tensor([0, random.choice(env['available_player_list']), 1])
                # 撤退干员
                elif i == 2:
                    action = torch.tensor([0, random.choice(env['available_player_list']), 1])
            return action, i
        else:
            # print("开始最优行为")
            # 【部署费用， 在场敌人数， 保卫点数】

            # 采用分散动作头（Mult-head）的输出方法，输出每个动作的q值再比较
            place_q, skill_q, retreat_q = self.q_net(state)
            """
            place_q[0][2][5] = [-0.7299, 1.1585, -0.2429, -0.3950]
            表示干员编号 2 放置在位置 5 的 4 个方向的 Q 值分别为：
            向上（0）：-0.7299
            向右（1）：1.1585
            向下（2）：-0.2429
            向左（3）：-0.3950
            最优方向为 向右，因为它的 Q 值最大（1.1585）。
            """
            # place_action = torch.argmax(place_q).item()  # 放置干员的动作
            # skill_action = torch.argmax(skill_q).item()  # 技能使用的动作
            # retreat_action = torch.argmax(retreat_q).item()  # 撤退的动作

            # 获取每种动作的最大 Q 值
            max_place_q = place_q.max().item()
            max_skill_q = skill_q.max().item()
            max_retreat_q = retreat_q.max().item()
            # print(f"放置q值{max_place_q},技能q值{max_skill_q}, 撤退q值{max_retreat_q}")

            # 比较最大 Q 值，选择动作类别
            # 放置
            if max_place_q >= max_skill_q and max_place_q >= max_retreat_q:
                # chosen_action = torch.argmax(place_q).item()  # 选择放置动作
                max_indices = torch.unravel_index(place_q.argmax(), place_q.shape)
                # print(f"Batch: {max_indices[0]}, Character: {max_indices[1]}, Position: {max_indices[2]}, Direction: {max_indices[3]}")
                return torch.tensor([max_indices[1], max_indices[2], max_indices[3]]), 0
            # 技能
            elif max_skill_q >= max_place_q and max_skill_q >= max_retreat_q:
                # chosen_action = torch.argmax(skill_q).item()  # 选择技能使用动作
                max_indices = torch.unravel_index(skill_q.argmax(), skill_q.shape)
                # print(f"Batch: {max_indices[0]}, Character: {max_indices[1]}, skill: {max_indices[2]}")
                return torch.tensor([0, max_indices[1], max_indices[2]]), 1
            # 撤退
            else:
                # chosen_action = torch.argmax(retreat_q).item()  # 选择撤退动作
                max_indices = torch.unravel_index(retreat_q.argmax(), retreat_q.shape)
                # print(f"Batch: {max_indices[0]}, Character: {max_indices[1]}, retreat: {max_indices[2]}")
                return torch.tensor([0, max_indices[1], max_indices[2]]), 2



    def update(self, transition_dict, arg:dict):
        gamma = arg['gamma']
        batch_size = arg['batch_size']

        states = torch.stack(transition_dict['states']).to(self.device)

        #会出现action数据长度不一致的情况RuntimeError: stack expects each tensor to be equal size, but got [4] at entry 0 and [3] at entry 1
        actions = torch.stack(transition_dict['actions']).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.stack(transition_dict['next_states']).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # q_values = self.q_net(states).gather(1, actions)  # Q值
        place_q, skill_q, retreat_q = self.q_net(states)
        # print(f"actions:{actions}")

        # 这里的2是batchsize
        current_place_q = place_q[torch.arange(batch_size), actions[:, 0], actions[:, 1], actions[:, 2]]
        # current_skill_q = skill_q[torch.arange(batch_size), actions[:, 3], actions[:, 4]]
        # current_retreat_q = retreat_q[torch.arange(batch_size), actions[:, 5], actions[:, 6]]

        # 下个状态的最大Q值
        with torch.no_grad():
            next_place_q, next_skill_q, next_retreat_q = self.target_q_net(next_states)
            # print(f"next_place_q:{next_place_q}")
            # 目标值
            # 对 next_place_q 的所有动作组合取最大值
            # next_place_q_max = next_place_q.view(4, -1).max(dim=1)[0]  # (batch_size,)

            # target_place_q = rewards + gamma * next_place_q_max * (1 - dones)
            target_place_q = rewards + gamma * next_place_q.max().item()
            # target_skill_q = rewards + gamma * next_skill_q.max().item()
            # target_retreat_q = rewards + gamma * next_retreat_q.max().item()

            # target_place_q = rewards + gamma * next_place_q.max(dim=1)[0] * (1 - dones)
            # target_skill_q = rewards + gamma * next_skill_q.max(dim=1)[0] * (1 - dones)
            # target_retreat_q = rewards + gamma * next_retreat_q.max(dim=1)[0] * (1 - dones)

        # 计算每个动作头的损失
        loss_place = nn.MSELoss()(current_place_q, target_place_q)
        # loss_skill = nn.MSELoss()(current_skill_q, target_skill_q)
        # loss_retreat = nn.MSELoss()(current_retreat_q, target_retreat_q)

        # 总损失
        total_loss = loss_place
        # total_loss = loss_place + loss_skill + loss_retreat

        # 反向传播更新网络
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                    self.q_net.state_dict())
        self.count += 1

        return total_loss.item()

        # max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
        #     -1, 1)
        # q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
        #                                                         )  # TD误差目标
        # dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        # self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        # dqn_loss.backward()  # 反向传播更新参数
        # self.optimizer.step()

if __name__ == '__main__':
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    # 经验回放池的最低训练阈值
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env = ArknightEnv()
    num_characters, num_positions, num_directions = env.action_space.spaces[0].nvec
    state_dim = len(env.observation_space.nvec) # 对应 MultiDiscrete 的维度数量
    agent = DQN(state_dim, hidden_dim, lr, gamma, epsilon,
                target_update, device)
    # action = agent.take_action(env)
    print(agent.q_net(torch.tensor([2,2,3], dtype=torch.float32).to('cuda')))

    # print(action)