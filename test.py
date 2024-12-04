import pyautogui
import numpy as np
from gym import spaces
import cv2
import ultralytics
import gym
import torch
import torch.nn.functional as F
import torch.nn as nn

num_characters = 8
num_positions = 10
num_directions = 4

class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_characters, num_positions, num_directions):
        super(Qnet, self).__init__()

        # 输入层：处理 observation_space 的维度
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # 隐藏层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 输出层：对应 action_space 的 3 个部分
        # 放置干员：[num_characters, num_positions, num_directions]
        self.place_layer = nn.Linear(hidden_dim, num_characters * num_positions * num_directions)

        # 技能使用：[num_characters, 2]
        self.skill_layer = nn.Linear(hidden_dim, num_characters * 2)

        # 撤退：[num_characters, 2]
        self.retreat_layer = nn.Linear(hidden_dim, num_characters * 2)

    def forward(self, x):
        # 前向传播：状态输入
        x = F.relu(self.fc1(x))  # 输入层
        x = F.relu(self.fc2(x))  # 隐藏层

        # 分别计算 3 个动作的 Q 值
        place_q = self.place_layer(x)
        skill_q = self.skill_layer(x)
        retreat_q = self.retreat_layer(x)

        # 调整输出形状
        place_q = place_q.view(-1, num_characters, num_positions, num_directions)
        skill_q = skill_q.view(-1, num_characters, 2)
        retreat_q = retreat_q.view(-1, num_characters, 2)

        return place_q, skill_q, retreat_q

observation_space = spaces.MultiDiscrete([99, 20, 4])
action_space = spaces.Tuple((
    spaces.MultiDiscrete([num_characters, num_positions, num_directions]),  # 放置干员
    spaces.MultiDiscrete([num_characters, 2]),  # 技能使用
    spaces.MultiDiscrete([num_characters, 2])  # 撤退
))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
# [在场敌人数、保卫点数、部署费用]
state = torch.tensor([5, 10, 2], dtype=torch.float32).to(device)
state_dim = len(observation_space.nvec)  # 对应 MultiDiscrete 的维度数量
num_characters, num_positions, num_directions = action_space.spaces[0].nvec
hidden_dim = 128

QNet = Qnet(state_dim, hidden_dim, num_characters, num_positions, num_directions).to(device)

place_q, skill_q, retreat_q = QNet(state)
place_action = torch.argmax(place_q).item()  # 放置干员的动作
skill_action = torch.argmax(skill_q).item()  # 技能使用的动作
retreat_action = torch.argmax(retreat_q).item()  # 撤退的动作

# 获取每种动作的最大 Q 值
max_place_q = place_q.max().item()
max_skill_q = skill_q.max().item()
max_retreat_q = retreat_q.max().item()

# 比较最大 Q 值，选择动作类别
if max_place_q >= max_skill_q and max_place_q >= max_retreat_q:
    action_type = "place"
    # chosen_action = torch.argmax(place_q).item()  # 选择放置动作
    max_indices = torch.unravel_index(place_q.argmax(), place_q.shape)
    print(f"Batch: {max_indices[0]}, Character: {max_indices[1]}, Position: {max_indices[2]}, Direction: {max_indices[3]}")

elif max_skill_q >= max_place_q and max_skill_q >= max_retreat_q:
    action_type = "skill"
    # chosen_action = torch.argmax(skill_q).item()  # 选择技能使用动作
    max_indices = torch.unravel_index(skill_q.argmax(), skill_q.shape)
    print(f"Batch: {max_indices[0]}, Character: {max_indices[1]}, skill: {max_indices[2]}")

else:
    action_type = "retreat"
    # chosen_action = torch.argmax(retreat_q).item()  # 选择撤退动作
    max_indices = torch.unravel_index(retreat_q.argmax(), retreat_q.shape)
    print(f"Batch: {max_indices[0]}, Character: {max_indices[1]}, retreat: {max_indices[2]}")
# max_indices = torch.unravel_index(place_q.argmax(), place_q.shape)
# print(
#     f"Batch: {max_indices[0]}, Character: {max_indices[1]}, Position: {max_indices[2]}, Direction: {max_indices[3]}")

# print(place_q, retreat_q, skill_q)

# 测试操作空间取值
# char = 12
# floor = 12
# dirt = 4
# space = spaces.MultiDiscrete([char, floor, dirt])
# x = space.sample()

# image = cv2.imread('./detect_image/screen.png')
# cropped_image = image[128:150, 128:160]
# cv2.imshow('Image', cropped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

env = gym.make('CartPole-v0')
hidden_dim = 128
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
lr = 2e-3
gamma = 0.98
epsilon = 0.01
target_update = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

QNet = Qnet(state_dim, hidden_dim,action_dim).to(device)
state = env.reset()
state = torch.tensor([state[0]], dtype=torch.float).to(device)
res = QNet(state)
"""
# action = torch.randint(0, torch.tensor([10,15,4]), (1,))
# print(action)

# position_list = [{'id': '桃金娘', 'position': (1,2), 'ditection': 'DOWN'}]
# print([a.get('id') for a in position_list])

# model = ultralytics.YOLO("model/train3.pt")
# res = model("res_image/shot_67.png")
# for r in res:
#     # 每个检测框的类别
#     print(r.boxes.cls)
#     # 每个检测框的置信度
#     print(r.boxes.conf)


# print(env.observation_space.high)
# print(env.observation_space.low)

# state = env.reset()
print()

