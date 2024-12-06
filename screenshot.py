import pyautogui
import time
import cv2
import pytesseract
from enum import Enum
import numpy as np
import re
from ultralytics import YOLO


class Cutter:
    def __init__(self):
        self.screen_parm = (0, 0, 1858, 1096)
        self.mumu_fee_roi_coords = (1740, 700, 2000, 860)  # 定义 ROI 区域，格式为 (x_start, y_start, x_end, y_end)
        self.phone_fee_roi_coords = (2550, 800, 2700, 960)

        self.mumu_point_roi_coords = (1010, 0, 1200, 200)
        self.phone_point_roi_coords = (0, 0, 1000, 1400)

        # 横向大概25， 纵向130， 蓝门前【1437， 400】
        self.mumu_enemy_point_coords = ()
        self.phone_enemy_point_coords = ()
        self.mumu_point_coords = ()
        self.phone_point_coords = ()

    class ScreenType(Enum):
        PHONE = 1
        PC = 2

    # 单次识别并识别敌人数(上方的)
    def image_enemy_point_detect(self, type: ScreenType):
        print("开始")
        roi_coords = self.mumu_enemy_point_coords if type == self.ScreenType.PC else self.phone_enemy_point_coords
        image = pyautogui.screenshot(region=self.screen_parm)
        binary_image = self.preprocess_image(image, roi_coords)
        recognized_numbers = self.segment_and_recognize(binary_image)
        number_list = re.findall(r'\d+', recognized_numbers)
        if len(number_list) != 0:
            recognized_numbers = re.findall(r'\d+', recognized_numbers)[0]
            print(recognized_numbers)
            return recognized_numbers
        else:
            print("Not Found.")

    def point_number_detect(self, image, type: ScreenType) -> int:
        roi_coords = self.mumu_point_roi_coords if type == self.ScreenType.PC else self.phone_fee_roi_coords
        # image = pyautogui.screenshot(region=self.screen_parm)
        binary_image = self.preprocess_image(image, roi_coords)
        recognized_numbers = self.segment_and_recognize(binary_image)
        number_list = re.findall(r'\d+', recognized_numbers)
        if len(number_list) != 0:
            recognized_numbers = re.findall(r'\d+', recognized_numbers)[0]
            return int(recognized_numbers)
        else:
            return -1

    # 连续截屏并识别
    def image_stream_enemy_detect(self, model:YOLO, type: ScreenType):
        # 对单张图片而言
        print("开始识别敌人")
        # roi_coords = self.mumu_fee_roi_coords if type == self.ScreenType.PC else self.phone_fee_roi_coords
        while True:
            image = pyautogui.screenshot(region=self.screen_parm)
            res = model(image)[0]
            print(f"模型识别到的类:{res.boxes.cls}")
            print(f"识别类的置信度：{res.boxes.conf}")

    def enemy_detect(self, image, model:YOLO, type: ScreenType):
        # image = pyautogui.screenshot(region=self.screen_parm)
        # image = cv2.imread('res_image/number.png')
        res = model(image)[0]
        return res

    # 连续截屏
    def image_stream_shot(self):
        print("开始截屏")
        num = 1
        while True:
            pyautogui.screenshot(imageFilename=f"./res_image/shot_{num}.png", region=self.screen_parm)
            num += 1
            time.sleep(1)

    # 连续识别数字
    def image_stream_number_detect(self, type: ScreenType):
        print("开始截屏")
        roi_coords = self.mumu_fee_roi_coords if type == self.ScreenType.PC else self.phone_fee_roi_coords
        while True:
            image = pyautogui.screenshot(region=self.screen_parm)
            binary_image = self.preprocess_image(image, roi_coords)
            recognized_numbers = self.segment_and_recognize(binary_image)
            number_list = re.findall(r'\d+', recognized_numbers)
            if len(number_list) != 0:
                recognized_numbers = re.findall(r'\d+', recognized_numbers)[0]
                print(recognized_numbers)
            else:
                print("Not Found.")
            time.sleep(1)

    # 单次识别部署费用
    def fee_number_detect(self, image, type: ScreenType) -> int:
        roi_coords = self.mumu_fee_roi_coords if type == self.ScreenType.PC else self.phone_fee_roi_coords
        # image = pyautogui.screenshot(region=self.screen_parm)
        binary_image = self.preprocess_image(image, roi_coords)
        recognized_numbers = self.segment_and_recognize(binary_image)
        number_list = re.findall(r'\d+', recognized_numbers)
        if len(number_list) != 0:
            recognized_numbers = re.findall(r'\d+', recognized_numbers)[0]
            return int(recognized_numbers)
        else:
            return -1

    def image_transfer(self):
        pass

    @staticmethod
    def preprocess_image(image, roi_coords=None):
        if isinstance(image, str):
            # 读取图像并转换为灰度
            image = cv2.imread(image)
        else:
            image = np.array(image)

        # 如果指定了 ROI 区域，则裁剪图像
        if roi_coords:
            x_start, y_start, x_end, y_end = roi_coords
            image = image[y_start:y_end, x_start:x_end]

            # cv2.imshow("Cropped ROI", image)
            # cv2.waitKey(0)  # 等待按键以关闭窗口
            # cv2.destroyAllWindows()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def segment_and_recognize(binary_image):
        denoised_image = cv2.medianBlur(binary_image, 3)

        # cv2.imshow("11", denoised_image)
        # cv2.waitKey(0)  # 等待按键以关闭窗口
        # cv2.destroyAllWindows()

        res = pytesseract.image_to_string(denoised_image, config='--psm 6')
        # # 查找轮廓
        # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # recognized_numbers = []
        #
        # for cnt in contours:
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     # 筛选合理大小的轮廓
        #     if 10 < w < 100 and 10 < h < 100:  # 假设数字的宽高在此范围内
        #         roi = binary_image[y:y + h, x:x + w]
        #         # 使用 OCR 识别数字
        #         config = "--psm 10 -c tessedit_char_whitelist=0123456789"
        #         number = pytesseract.image_to_string(roi, config=config).strip()
        #         recognized_numbers.append(number)
        #
        #         # 可视化识别
        #         cv2.rectangle(binary_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #         cv2.putText(binary_image, number, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return res


if __name__ == '__main__':
    cutter = Cutter()

    # cutter.image_stream_shot()
    model = YOLO("model/train3.pt")
    # cutter.image_stream_enemy_detect(model, Cutter.ScreenType.PC)

    # print(cutter.point_number_detect(Cutter.ScreenType.PC))
    # image = pyautogui.screenshot(region=cutter.screen_parm)
    image = cv2.imread('res_image/number.png')
    res = cutter.enemy_detect(image, model, Cutter.ScreenType.PC)

    print(res.names.get(res.boxes.cls.item()), len(res.boxes))
    # image_path = 'res_image/number.png'
    # mumu_roi_coords = (1740, 700, 2000, 860)  # 定义 ROI 区域，格式为 (x_start, y_start, x_end, y_end)
    # phone_roi_coords = (2550, 800, 2700, 960)
    #
    # binary_image = cutter.preprocess_image(image_path, mumu_roi_coords)
    # recognized_numbers = cutter.segment_and_recognize(binary_image)
    #
    # print("识别出的数字:", recognized_numbers)
