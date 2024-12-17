import json
import os
import time
from enum import Enum
from typing import Tuple, Optional, Union, List

import cv2
import numpy as np
import pyautogui
import requests
import torch
from gym import Env, spaces
from gym.core import RenderFrame
from torch import Tensor
from ultralytics import YOLO

from screenshot import Cutter


class ArknightEnv(Env):

    # TODO:完成observation_space观察空间的设计(暂定为【在场敌人数、保卫点数、部署费用】，后期可以考虑加的参数：一个锚定敌人与蓝门距离的参数、总敌人数等)

    class ActType(Enum):
        PLACE = 0
        SKILL = 1
        REMOVE = 2
        WAIT = 4

    class DirectionType(Enum):
        LEFT = 0
        UP = 1
        RIGHT = 2
        DOWN = 3

    """
    - :action_space可以被划分成三个MultiDiscrete
        1) 放置干员: MultiDiscrete [干员数(n), 可放置方块数(k), 朝向(4)]
        - 干员数 - params: min: 0, max: 12
        - 可放置方块数 - params: min: 0, max: 12
        - 朝向 - params: min: 0, max: 4

        2) 技能使用: MultiDiscrete [干员数 (n), 是否使用技能(2)]
        - 干员数 - params: min: 0, max: 12
        - 是否使用技能 - params: min: 0, max: 1

        3) 撤退干员: MultiDiscrete [干员数(n), 是否撤退干员(2)]
        - 干员数 - params: min: 0, max: 12
        - 是否撤退干员 - params: min: 0, max: 1
    """

    """
    NOTE: !!只需要更新fee、position_list、player_freeze_list就可以自动更新available_player_list!!
    """

    # 进入部署冷却状态的干员
    @property
    def freeze_player_list_id(self):
        return [i for i, b in enumerate(self.player_freeze_list) if b]

    # 根据fee进行自动更新
    @property
    def fee_available_player_list(self):
        return [b for a, b in zip(self.player_fee_list, self.player_list) if a <= self.fee]

    # 根据position_list和fee_available_player_list自动更新，可用干员列表
    @property
    def available_player_list(self):
        # 可放置干员列表
        return [self.player_list[i] for i in self.available_player_list_id]

    # 同上，是index版
    @property
    def available_player_list_id(self):
        # 可放置干员列表
        return [i for i, b in enumerate(self.player_status_list) if
                b and (self.player_list[i] in self.fee_available_player_list) and (i not in self.freeze_player_list_id)]

    @property
    def position_list_id(self):
        # 已放置干员列表index版
        return [i.get('id') for i in self.position_list]

    # 定义一个属性用来描述敌人距离蓝门的距离,假设距离蓝门的地块越近，那么该值越大越好
    @property
    def enemy_to_point(self):
        return 0

    @property
    # 可放置的方块列表
    def available_position_list(self):
        res = [i for i, b in enumerate(self.position_status_list) if b]
        return res

    def __init__(self):

        # 观测空间，暂定128*128大小
        # self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)

        self.enemy = 0
        self.point = 15
        # 拖动干员时出现的坐标偏移值
        self.transfer_distance_row = 100
        self.transfer_distance_col = -30

        self.skill_button_position = (1277, 677)
        self.remove_button_position = (860, 358)

        # 操作时间
        self.action_time = 1
        # 部署费用（初始值设置为99方便测试）
        self.fee = 0

        self.reward = 0
        self.done = False
        # 视觉处理
        self.cutter = Cutter()
        self.enemy_model = YOLO(os.path.join("model", "train3.pt"))

        # 在场干员列表,是相对于player_list的序号列表
        # self.position_list = [{'id': 6, 'position': (1,2), 'ditection': 'DOWN'}]
        self.position_list = []

        # TODO: 将干员信息配置成外部json文件
        with open("data.json", 'r', encoding='utf-8') as f:
            text = f.read()
            self.data = json.loads(text)

        self.player_fee_list = [i['fee'] for i in self.data['players'].values()]
        self.player_list = [i for i in self.data['players']]
        self.high_player_list = [name for name, player in self.data["players"].items() if player["ishigh"] == 1]

        self.num_characters = len(self.player_list)
        self.player_status_list = [True for _ in self.player_list]


        # 每个可放置地块的坐标列表，序号越小越靠近蓝门
        self.position_location_list = self.data['position_location_list']
        # 可放置方块数
        self.num_positions = len(self.position_location_list)
        # 方块可放置状态
        self.position_status_list = [True for _ in range(self.num_positions)]

        self.high_floor_list_id = self.data['high_floor_list_id']

        # 技能冷却列表,技能准备好的干员id列表
        self.skill_ready_list_id = []
        # 干员冷却状态
        self.player_freeze_list = [False for _ in self.player_list]
        # 朝向
        self.num_directions = 4

        # self.action_max_dim = (len(self.player_list), self.num_positions, 4)

        # 多重离散空间，其中一个动作可以通过sample()随机采取动作，选用tuple列举三个不同的动作，再从大的三个动作里随机选取
        self.action_space = spaces.Tuple((
            spaces.MultiDiscrete([self.num_characters, self.num_positions, self.num_directions]),  # 放置干员
            spaces.MultiDiscrete([self.num_characters, 2]),  # 技能使用
            spaces.MultiDiscrete([self.num_characters, 2]),
            spaces.MultiDiscrete([1, 1])  # 等待
        ))

        # 指的是离散值，即观察空间有多大，值的取值范围是多少【部署费用， 在场敌人数， 保卫点数】
        self.observation_space = spaces.MultiDiscrete([99, 20, 4])

    def update(self):
        image = pyautogui.screenshot(region=self.cutter.screen_parm)
        # self.cutter.image_stream_enemy_detect(Cutter.ScreenType.PC)
        # 更新识别到的数字
        if self.cutter.fee_number_detect(image, Cutter.ScreenType.PC) != -1:
            self.fee = self.cutter.fee_number_detect(image, Cutter.ScreenType.PC)

        # file_path = os.path.join("model", "train3.pt")
        self.enemy = len(self.cutter.enemy_detect(image, self.enemy_model, Cutter.ScreenType.PC).boxes)
        self.point = self.cutter.point_number_detect(image, Cutter.ScreenType.PC)
        print(f"费用：{self.fee}, 敌人：{self.enemy}，保卫点{self.point}")
        return self.cutter.end_detect(image)

    # 可视化处理
    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    # 根据输入动作进行推演，返回observation, reward, terminated, truncated, info
    def step(self, arg:Tuple) -> Tuple[Tensor, float, bool, bool, dict]:
        action, type = arg
        if type == 0:
            name = self.player_list[action[0]]
            position = self.position_location_list[action[1]]

            # FIXME：在修复干员放置错误的bug前先限制非法操作
            if (name in self.high_player_list and action[1] in self.high_floor_list_id) or (
                    name not in self.high_player_list and action[1] not in self.high_floor_list_id):
                self.place(name, position, self.action_time, ArknightEnv.DirectionType(action[2].item()))
            else:
                print(f"干员放置位置错误！空操作,放置干员{name}在{action[1]}、"
                      f"high_player_list：{self.high_player_list},"
                      f"high_floor_list_id:{self.high_floor_list_id}")

        elif type == 1:
            self.skill(self.player_list[action[1]])
        elif type == 2:
            self.remove(self.player_list[action[1]])
        elif type == 3:
            print("等待.")
            time.sleep(action[0])

        print(
            f"可放置方块:{self.available_position_list}, "
            f"可放置干员列表:{self.available_player_list}，"
            f"冷却干员id:{self.freeze_player_list_id},"
            f"干员冷却列表:{self.skill_ready_list_id}"
        )

        # 环境更新
        done = self.update()

        # 下一个state【部署费用， 在场敌人数， 保卫点数】
        state = torch.tensor([self.fee, self.enemy, self.point], dtype=torch.float32).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # 奖励函数
        reward = -0.2 * self.fee + -1 * self.enemy + 1 * self.point

        if done == True:
            print("游戏结束")
        truncated = False
        info = {}
        return state, reward, done, truncated, info

    # 环境重置,返回最初的状态
    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None,
              ):
        # TODO:重开游戏

        return torch.tensor([self.fee, 0, 0], dtype=torch.float32).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # TODO:扫描地块
    def scan_floor(self):
        pass
        return self.position_location_list

    def place(self, name, position, time, direction: DirectionType):
        id = self.player_list.index(name)
        left_top = (0, 0)
        # print(f"place前费用:{self.fee}")
        # print(f"place前可用干员列表:{self.available_player_list}")
        # print(f"place前已放置干员列表:{self.position_list}")
        try:
            player_bottom_card_list = [i for i in range(self.num_characters) if i not in self.position_list_id]
            # 在下方卡片里的序号
            transform_id = player_bottom_card_list.index(id)
        except ValueError as e:
            print(f"{name}不在可放置干员列表！操作失败,error:{e}")
            return

        # 这里不能直接用id了，因为放置干员后下面的干员列表会出现变化，要通过转化才能用
        # 选中干员位置
        start = (left_top[0] + 1760 + (-180 * transform_id), left_top[1] + 1010)
        pyautogui.moveTo(start[0], start[1])
        x, y = self.transform_site(position[0], position[1])
        pyautogui.dragTo(x, y, duration=time)

        if direction == ArknightEnv.DirectionType.LEFT:
            pyautogui.dragRel(-200, 0, duration=time)
        elif direction == ArknightEnv.DirectionType.UP:
            pyautogui.dragRel(0, -200, duration=time)
        elif direction == ArknightEnv.DirectionType.RIGHT:
            pyautogui.dragRel(200, 0, duration=time)
        elif direction == ArknightEnv.DirectionType.DOWN:
            pyautogui.dragRel(0, 200, duration=time)

        print(f"放置干员{name}在{self.position_location_list.index(position)},方向为{direction}")
        self.position_list.append({'id': id, 'position': position, 'ditection': direction.value})
        # self.available_player_list.remove(self.available_player_list[transform_id])
        self.player_status_list[id] = False
        self.position_status_list[id] = False

        # TODO：暂定干员放置后技能冷却完毕
        self.skill_ready_list_id.append(id)
        self.fee -= self.player_fee_list[self.player_list.index(name)]
        # print(f"place后费用:{self.fee}")
        # print(f"place后可部署干员:{self.available_player_list}")
        # print(f"place后已放置干员列表:{self.position_list}")

    def remove(self, name):
        id = self.player_list.index(name)
        target = next((item for item in self.position_list if item.get("id") == id), None)
        target_postion = target.get("position")
        pyautogui.click(target_postion[0], target_postion[1])
        pyautogui.click(self.remove_button_position[0], self.remove_button_position[1])

        # self.player_status_list[id] = True
        # FIXME:加入部署冷却机制，更新冷却队列
        # 放入部署冷却队列
        self.player_freeze_list[id] = True
        # 更新可放置方块列表
        self.position_status_list[id] = True
        # 更新已放置干员列表
        self.position_list = [i for i in self.position_list if i.get("id") != id]
        # 移出技能冷却列表(可能已经使用了技能)
        if id in self.skill_ready_list_id:
            self.skill_ready_list_id.remove(id)
        print(f"撤回干员{name}")
        # print(f"撤退后放置的干员：{self.available_player_list}")
        # print(f"撤退后的干员状态{self.player_status_list}")

    def skill(self, name):
        id = self.player_list.index(name)
        target = next((item for item in self.position_list if item.get("id") == id), None)
        target_postion = target.get("position")
        pyautogui.click(target_postion[0], target_postion[1])
        pyautogui.click(self.skill_button_position[0], self.skill_button_position[1])

        # 移出技能冷却列表
        if id in self.skill_ready_list_id:
            self.skill_ready_list_id.remove(id)
        print(f"干员技能使用{name}")


    def output_dic(self):
        res = {
            'available_position_list':self.available_position_list,
            'position_list':self.position_list,
            # 'action_max_dim':self.action_max_dim,
            'player_fee_list':self.player_fee_list,
            'fee':self.fee,
            'available_player_list_id':self.available_player_list_id,
            'position_list_id': self.position_list_id,
            'high_floor_list_id': self.high_floor_list_id,
            'high_player_list_id': [self.player_list.index(i) for i in self.high_player_list],
            'skill_ready_list_id': self.skill_ready_list_id
        }
        return res

    # 处理交互部分，获取action返回state,即一局游戏
    def client(self):
        url = 'http://127.0.0.1:6006/action'
        done = False
        while not done:
            # 模拟环境状态
            # 'position_list': self.position_list,
            state = {
                "state": [self.fee, self.enemy, self.point],
                "reward": self.reward, "done": True,
                "dic": self.output_dic()
            }
            # POST 请求发送状态数据
            response = requests.post(url, json=state)
            # print("Sending state:", state)

            data = response.json()
            # print("Received action:", data)

            next_state, reward, done, _, _ = self.step((torch.tensor(data.get("action")), data.get("type")))

    def transform_site(self, x, y):
        matrix = np.load("transform_matrix.npy")
        point = np.array([x, y, 1.0], dtype=np.float32)
        transformed = np.dot(matrix, point)
        tx, ty = transformed[0] / transformed[2], transformed[1] / transformed[2]
        return tx, ty

    def output_action_dim(self):
        return [self.num_characters, self.num_positions, self.num_directions]

if __name__ == '__main__':
    env = ArknightEnv()

    # floor_1 = (1306, 407)
    # platform_1 = (1155, 510)
    # floor_2 = (1155, 400)

    # env.place("泡普卡", env.position_location_list[2], 1, ArknightEnv.DirectionType.LEFT)
    # env.skill("泡普卡")

    position_location_list = [(1439, 405), (1420, 266), (1289, 392), (1471, 529),
                              (1281, 261), (1130, 389), (1321, 519), (1502, 671),
                              (1137, 267), (985, 379), (1160, 520), (1338, 653),
                              (1545, 823), (998, 263), (841, 392), (992, 518), (1172, 681), (1365, 819)]
    transfer_distance_row = 100
    transfer_distance_col = -30
    x, y = position_location_list[9]
    ox, oy = x + transfer_distance_row, y + transfer_distance_col

    # 偏移前的坐标
    src_points = np.float32([
        [426, 289], [575, 282], [714, 280], [856, 285], [999, 268], [1137, 265], [1284, 265], [1435, 261],
        [385, 385], [543, 369], [703, 378], [863, 385], [994, 389], [1156, 390], [1296, 392], [1145, 393],
        [325, 652], [495, 647], [678, 652], [840, 648], [1015, 665], [1166, 668], [1339, 668], [1504, 661],
        [305, 825], [485, 822], [656, 823], [841, 823], [1007, 814], [1193, 809], [1373, 814], [1546, 818],

    ])

    # 偏移后的坐标
    dst_points = np.float32([
        [612, 362], [734, 343], [852, 330], [980, 317], [1109, 286], [1243, 271], [1372, 254], [1518, 241],
        [587, 453], [709, 431], [845, 418], [985, 409], [1115, 402], [1258, 390], [1393, 374], [1547, 359],
        [552, 672], [692, 656], [839, 660], [979, 652], [1139, 649], [1280, 636], [1438, 623], [1605, 607],
        [471, 824], [693, 813], [832, 809], [985, 799], [1145, 783], [1310, 770], [1485, 754], [1658, 742],
    ])

    # 使用仿射变换拟合
    matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    np.save("transform_matrix.npy", matrix)

    # 测试映射函数
    # def transform_point(x, y, matrix):
    #     point = np.array([x, y, 1.0], dtype=np.float32)
    #     transformed = np.dot(matrix, point)
    #     tx, ty = transformed[0] / transformed[2], transformed[1] / transformed[2]
    #     return tx, ty

    # 测试示例
    x, y = position_location_list[8]
    x_new, y_new = env.transform_site(x, y)
    print(f"偏移后坐标: ({x_new:.2f}, {y_new:.2f})")
    pyautogui.moveTo(x_new, y_new)

    # pyautogui.dragTo(ox, oy)
