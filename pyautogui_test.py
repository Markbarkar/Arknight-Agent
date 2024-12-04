import pyautogui
import time
num_seconds= 1

windows_image = "./detect_image/screen.png"
windows_location = pyautogui.locateOnScreen(windows_image, grayscale=False, confidence=0.8)
print(f"窗口识别坐标：{windows_location}")

game_flag = True
start_button_location = pyautogui.locateOnScreen("./detect_image/start_game_button.png", grayscale=False, confidence=0.4)
if start_button_location is not None:
    print("发现游戏开始键")
    pyautogui.mouseDown(x=start_button_location[0] + 100, y=start_button_location[1] + 100, button="left")
    pyautogui.mouseUp()
accomplished_flag = pyautogui.locateOnScreen("./detect_image/accomplished_sign.png", grayscale=False, confidence=0.6)

while accomplished_flag is None:
    bug_location = pyautogui.locateOnScreen("./detect_image/bug.png", grayscale=False, confidence=0.5)
    dog_location = pyautogui.locateOnScreen("./detect_image/dog.png", grayscale=False, confidence=0.6)
    avaliable_floor_location = pyautogui.locateAllOnScreen("./detect_image/avaliable_floor.png", grayscale=False, confidence=0.6)
    enemy_entry_location = pyautogui.locateOnScreen("./detect_image/enemy_entry.png", grayscale=False, confidence=0.5)

    if bug_location is not None:
        pyautogui.moveTo(bug_location[0], bug_location[1], duration=0.5)
        # pyautogui.screenshot(imageFilename='./res_image/bug_result.jpg', region=bug_location)
    if dog_location is not None:
        pyautogui.moveTo(dog_location[0], dog_location[1], duration=0.5)
        # pyautogui.screenshot(imageFilename='./res_image/dog_result.jpg', region=dog_location)

    time.sleep(2)
    print(f"源石虫：{bug_location}\n 狗：{dog_location}\n 可放置方块：{avaliable_floor_location}\n 出怪点:{enemy_entry_location}")

print("是否结束？" + str(accomplished_flag))