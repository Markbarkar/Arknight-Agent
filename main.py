from Env import ArknightEnv
from time import sleep
from screenshot import Cutter
import keyboard
import torch
import torch.nn.functional as F
from agent import DQN,ReplayBuffer
from tqdm import tqdm

floor_1 = (1306, 407)
platform_1 = (1155, 510)
floor_2 = (1155, 400)

env = ArknightEnv()
cutter = Cutter()
done = False
lr = 2e-3
# 训练轮数，也就是一局（500）
num_episodes = 50
# 隐藏层大小
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
# 经验回放池的最低训练阈值(500)
minimal_size = 5
# 经验回放池随机抽取训练数（64）
batch_size = 2
replay_buffer = ReplayBuffer(buffer_size)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

num_characters, num_positions, num_directions = env.action_space.spaces[0].nvec
state_dim = len(env.observation_space.nvec)  # 对应 MultiDiscrete 的维度数量
agent = DQN(state_dim, hidden_dim, lr, gamma, epsilon,
            target_update, device)

for i in range(10):
    # 用来显示进度条的,把一整个训练过程分成10段，总训练次数还是num_episodes
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            # TODO: 环境的重置
            state = env.reset()
            done = False
            while not done:
                # TODO: 考虑要不要修改为传递state
                action = agent.take_action(state, env.output_dic())
                tqdm.write(f"action:{str(action)}")
                # TODO: 环境交互 next_state, reward = env.step(action)
                next_state, reward, done, _, _ = env.step(action)

                # 这里的done存储用浮点数，更新网络计算q值的时候结束时q为0，要用到done
                replay_buffer.add(state, action, reward, next_state, float(done))
                # TODO: 更新状态和奖励 state = next_state, episode_return += reward

                # 当经验回放池buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'rewards': b_r,
                        'next_states': b_ns,
                        'dones': b_d
                    }
                    # print(transition_dict)
                    agent.update(transition_dict)

                # number = int(cutter.image_number_detect(Cutter.ScreenType.PC))
                # print(number)
                # if number > 14 and '芬' in env.available_player_list:
                #     env.place("芬", floor_2, 0.7, ArknightEnv.DirectionType.LEFT)
                #     sleep(1)
                # if number > 8 and '桃金娘' in env.available_player_list:
                #     env.place("桃金娘", floor_1, 0.7, ArknightEnv.DirectionType.DOWN)
                #     sleep(1)
                # sleep(1)
            pbar.update(1)

# env.place("桃金娘", floor_1, 0.7, ArknightEnv.DirectionType.LEFT)
# sleep(15)
# env.skill("桃金娘")
# env.place("芬", floor_2, 0.7, ArknightEnv.DirectionType.LEFT)
# sleep(10)
# env.place("克洛斯", platform_1, 0.7, ArknightEnv.DirectionType.DOWN)
# env.remove("桃金娘")