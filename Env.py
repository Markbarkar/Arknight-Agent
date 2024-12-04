import random
from typing import Tuple, Optional, Union, List
import pyautogui
import numpy as np
import torch
from gym import Env, spaces
from gym.core import ObsType, RenderFrame
from enum import Enum
from time import sleep
from torch import Tensor
from ultralytics import YOLO

from screenshot import Cutter


class ArknightEnv(Env):

    # TODO:完成observation_space观察空间的设计(暂定为【在场敌人数、保卫点数、部署费用】，后期可以考虑加的参数：一个锚定敌人与蓝门距离的参数、总敌人数等)
    # TODO: 完善干员的索引，是用名字来输入action还是用id

    class ActType(Enum):
        PLACE = 1
        SKILL = 2
        REMOVE = 3

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
    NOTE: !!只需要更新fee和position_list就可以自动更新available_player_list!!
    """
    # 根据fee进行自动更新
    @property
    def fee_available_player_list(self):
        return [b for a, b in zip(self.player_fee_list, self.player_list) if a <= self.fee]

    # 根据position_list和fee_available_player_list自动更新
    @property
    def available_player_list(self):
        id_list = [item.get('id') for item in self.position_list if 'id' in item]
        return [b for b in self.fee_available_player_list if b not in id_list]

    # 定义一个属性用来描述敌人距离蓝门的距离,假设距离蓝门的地块越近，那么该值越大越好
    @property
    def enemy_to_point(self):
        return 0

    # @property
    # def observation_space(self):
    #     # [部署费用， 在场敌人数量， 保卫点数]
    #     return spaces.MultiDiscrete([self.fee, self.enemy, self.point])

    def __init__(self):

        # 观测空间，暂定128*128大小
        # self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)

        self.enemy = 0
        self.point = 15
        # 拖动干员时出现的坐标偏移值
        self.transfer_distance_row = 100
        self.transfer_distance_col = -30

        # 选择干员时出现的坐标偏移值
        self.select_distance_row = -250
        self.select_distance_col = 100

        # TODO: 将干员信息配置成外部json文件
        self.player_fee_list = [25, 17, 16, 16, 16, 9, 8, 9]

        # 每个可放置地块的坐标列表，序号越小越靠近蓝门
        self.position_location_list = [(1439, 405), (1420, 266), (1289, 392), (1471, 529),
               (1281, 261), (1130, 389), (1321, 519), (1502, 671),
               (1137, 267), (985, 379), (1160, 520), (1338, 653),
               (1545, 823), (998, 263), (841, 392), (992, 518), (1172, 681), (1365, 819)]

        self.player_list = ['维什戴尔','泡普卡', '史都华德', '苏苏洛', '卡提', '克洛斯', '桃金娘', '芬']
        self.caculate_list = [item for item in range(len(self.player_list))]

        # 部署费用
        self.fee = 99

        # 视觉处理
        self.cutter = Cutter()

        # 在场干员列表
        # self.position_list = [{'id': '桃金娘', 'position': (1,2), 'ditection': 'DOWN'}]
        self.position_list = []

        # 可放置的方块列表
        self.available_position_list = [1,2,3,4,5,6,7]
        # 干员数
        self.num_characters = len(self.player_list)
        # 可放置方块数
        self.num_positions = len(self.available_position_list)
        # 朝向
        self.num_directions = 4
        self.action_max_dim = (len(self.player_list), self.num_positions, 4)

        # 多重离散空间，其中一个动作可以通过sample()随机采取动作，选用tuple列举三个不同的动作，再从大的三个动作里随机选取
        self.action_space = spaces.Tuple((
            spaces.MultiDiscrete([self.num_characters, self.num_positions, self.num_directions]),  # 放置干员
            spaces.MultiDiscrete([self.num_characters, 2]),  # 技能使用
            spaces.MultiDiscrete([self.num_characters, 2])  # 撤退
        ))

        # 指的是离散值，即观察空间有多大，值的取值范围是多少【部署费用， 在场敌人数， 保卫点数】
        self.observation_space = spaces.MultiDiscrete([99, 20, 4])

    def update(self):
        # self.cutter.image_stream_enemy_detect(Cutter.ScreenType.PC)
        self.fee = self.cutter.fee_number_detect(Cutter.ScreenType.PC)
        self.enemy = self.cutter.enemy_detect(YOLO("model/train3.pt"), Cutter.ScreenType.PC)
        self.point = self.cutter.point_number_detect(Cutter.ScreenType.PC)

    # 可视化处理
    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    # 根据输入动作进行推演，返回observation, reward, terminated, truncated, info
    def step(self, action: Tensor) -> Tuple[Tensor, float, bool, bool, dict]:

        # TODO:完善step步骤, 添加检测,包括费用不足/干员已放置/干员未放置/地块已放置/地块不可放置
        if type == 0:
            self.place(self.player_list[action[0]], self.position_location_list[action[1]], 0.7, ArknightEnv.DirectionType(action[2].item()))
        elif type == 1:
            self.skill(self.player_list[action[0]])
        elif type == 2:
            self.remove(self.player_list[action[0]])

        # 下一个state【部署费用， 在场敌人数， 保卫点数】
        # state = torch.tensor([self.fee, self.enemy, self.point])
        state = torch.rand((3, ),) * torch.tensor([10, 3, 3])
        state = state.round().int()
        reward = -1.0 + random.random() * 2
        done = True

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

        return torch.tensor([0, 0, 0], dtype=torch.float32).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def scan_floor(self):
        self.position_location_list = [(1439, 405), (1420, 266), (1289, 392), (1471, 529),
               (1281, 261), (1130, 389), (1321, 519), (1502, 671),
               (1137, 267), (985, 379), (1160, 520), (1338, 653),
               (1545, 823), (998, 263), (841, 392), (992, 518), (1172, 681), (1365, 819)]
        return self.position_location_list

    def place(self, name, position, time, direction: DirectionType):
        id = self.player_list.index(name)
        left_top = (0, 0)
        # 在可选干员里的序号
        transform_id = self.available_player_list.index(self.player_list[id])

        # 这里不能直接用id了，因为放置干员后下面的干员列表会出现变化，要通过转化才能用
        start = (left_top[0] + 1760 + (-180 * transform_id), left_top[1] + 1010)
        pyautogui.moveTo(start[0], start[1])
        pyautogui.dragTo(position[0] + self.transfer_distance_row, position[1] + self.transfer_distance_col, duration=time)

        if direction == ArknightEnv.DirectionType.LEFT:
            pyautogui.dragRel(-200, 0, duration=time)
        elif direction == ArknightEnv.DirectionType.UP:
            pyautogui.dragRel(0, -200, duration=time)
        elif direction == ArknightEnv.DirectionType.RIGHT:
            pyautogui.dragRel(200, 0, duration=time)
        elif direction == ArknightEnv.DirectionType.DOWN:
            pyautogui.dragRel(0, 200, duration=time)

        print(f"放置干员{name}在{position},方向为{direction}")
        self.position_list.append({'id': id, 'position': position, 'ditection': direction})
        self.available_player_list.remove(self.available_player_list[transform_id])
        self.caculate_list.remove(id)
        self.fee -= self.player_fee_list[self.player_list.index(name)]
        print(f"更新部署费用:{self.fee}")
        print(f"目前可部署干员:{self.available_player_list}")
        # print(self.caculate_list)

    def remove(self, name):
        id = self.player_list.index(name)
        target = next((item for item in self.position_list if item.get("id") == id), None)
        target_postion = target.get("position")
        pyautogui.click(target_postion[0], target_postion[1])
        # 这里加了选择干员时的偏移值，先保证鼠标锚定人物中心
        pyautogui.moveTo(target_postion[0] + self.select_distance_row, target_postion[1] + self.select_distance_col)
        # 这里是撤退按钮相对于干员中心的偏移值
        pyautogui.moveRel(-150, -150)
        pyautogui.click()

        # 根据name计算插入到待定干员的序列（保证顺序）
        i = 0
        while i < len(self.caculate_list) and id > self.caculate_list[i]:
            i += 1
        self.available_player_list.insert(id, name)
        self.caculate_list.insert(i, id)
        print(f"撤回干员{name}")
        print(self.available_player_list)
        print(self.caculate_list)

    def skill(self, name):
        id = self.player_list.index(name)
        target = next((item for item in self.position_list if item.get("id") == id), None)
        target_postion = target.get("position")
        pyautogui.click(target_postion[0], target_postion[1])
        pyautogui.moveTo(target_postion[0] + self.select_distance_row, target_postion[1] + self.select_distance_col)
        pyautogui.moveRel(200, 170)
        pyautogui.click()

    def output_dic(self):
        res = {
            'available_position_list':self.available_position_list,
            'position_list':self.position_list,
            'action_max_dim':self.action_max_dim
        }
        return res


if __name__ == '__main__':
    env = ArknightEnv()

    floor_1 = (1306, 407)
    platform_1 = (1155, 510)
    floor_2 = (1155, 400)
    # 测试发现0.5s的操作似乎会拖动中断
    env.place("桃金娘", floor_1, 0.7, ArknightEnv.DirectionType.LEFT)
    sleep(15)
    env.skill("桃金娘")
    env.place("芬", floor_2, 0.7, ArknightEnv.DirectionType.LEFT)
    sleep(10)
    env.place("克洛斯", platform_1, 0.7, ArknightEnv.DirectionType.DOWN)
    env.remove("桃金娘")
