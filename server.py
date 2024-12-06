from flask import Flask, request, jsonify
from agent import DQN, ReplayBuffer
import torch
from queue import Queue
import threading
import time
import random
from gym import Env, spaces

app = Flask(__name__)

class server():

    def __init__(self):
        # 全局队列，用于传递输入数据
        self.request_queue = Queue()
        self.response_queue = Queue()

        self.agent = None
        self.replay_buffer = None
        self.hidden_dim = 128
        self.lr = 2e-3
        self.gamma = 0.98
        self.epsilon = 0.2
        self.target_update = 10
        self.batch_size = 3
        self.buffer_size = 10000

        self.observation_space = spaces.MultiDiscrete([99, 20, 4])

        # self.env = ArknightEnv()
        self.state_dim = len(self.observation_space.nvec)  # 对应 MultiDiscrete 的维度数量

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.agent = DQN(self.state_dim, self.hidden_dim, self.lr, self.gamma, self.epsilon,
                    self.target_update, self.device)
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.minimal_size = 5

    def model_worker(self, agent, replay_buffer):
        stop = False
        while not stop:
            done = False
            last_action = None
            last_state = None
            # 每一局的总reward
            episode_return = 0
            while not done:
                data = self.request_queue.get()

                state, reward, done, env_dic = data.get("state"), data.get("reward"), data.get("done"), data.get("dic")
                print(f"Received state: {state}, reward: {reward}, done:{done}")

                episode_return += reward

                action, type = agent.take_action(torch.tensor(state, dtype=torch.float32).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")), env_dic)

                if last_action is not None and last_state is not None:
                    replay_buffer.add(last_state, action, reward, state, float(done))

                if replay_buffer.size() > self.minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(self.batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'rewards': b_r,
                        'next_states': b_ns,
                        'dones': b_d
                    }
                    agent.update(transition_dict, {'gamma': self.gamma, 'batch_size': self.batch_size})

                last_action = action
                last_state = state
                print(f"Generated action: {action}")

                # 将结果放入响应队列
                self.response_queue.put((action, type))

Server = server()

@app.route('/action', methods=['POST'])
def get_action():
    global Server
    # 从客户端接收数据
    data = request.json
    state = data.get("state", {})
    reward = data.get("reward", 0)
    done = data.get("done", True)
    dic = data.get("dic", {})

    # 将数据放入请求队列
    Server.request_queue.put({"state": state, "reward": reward, "done": done, "dic": dic})
    print(f"state:{state}, reward:{reward}, done: {done}, dic: {dic}")

    # 等待响应队列中的结果（阻塞）
    action, type = Server.response_queue.get()
    print(f"action:{action}")
    return jsonify({"action": action.tolist(), "type": type})


    # 经验回放池随机抽取训练数（64）
    # num_characters, num_positions, num_directions = env.action_space.spaces[0].nvec

if __name__ == "__main__":
    # 启动后台线程
    threading.Thread(target=Server.model_worker, args=(Server.agent, Server.replay_buffer), daemon=True).start()

    app.run(host="0.0.0.0", port=6006)
