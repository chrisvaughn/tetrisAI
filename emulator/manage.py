import os
import subprocess
import time

import applescript
import cv2

from bot.detect import image_path
from emulator import capture, keyboard

EMULATOR_NAME = "Nestopia"
EMULATOR_PATH = "/Applications/Nestopia.app/Contents/MacOS/Nestopia"
ROM_PATH = "../ROMS/Tetris.nes"

dir_path = os.path.dirname(os.path.realpath(__file__))


def enter_speed_cheat():
    r = applescript.run(os.path.join(dir_path, "enter_nestopia_cheat.scpt"))
    return r.code == 0


def music_off(pid: int):
    print("Turning Music Off")
    for _ in range(3):
        keyboard.send_event(pid, "move_down", hold=0.5)
        time.sleep(0.5)


def select_level(pid: int, level: int):
    print(f"Selecting Level {level}")
    if level < 5:
        for _ in range(level):
            keyboard.send_event(pid, "move_right", hold=0.5)
            time.sleep(0.5)
    elif level < 10:
        keyboard.send_event(pid, "move_down", hold=0.5)
        for _ in range(level - 5):
            keyboard.send_event(pid, "move_right", hold=0.5)
            time.sleep(0.5)

    if level < 10:
        keyboard.send_event(pid, "return", hold=0.5)
        return

    if level < 15:
        for _ in range(level - 10):
            keyboard.send_event(pid, "move_right", hold=0.5)
            time.sleep(0.5)
    elif level < 20:
        keyboard.send_event(pid, "move_down", hold=0.5)
        for _ in range(level - 15):
            keyboard.send_event(pid, "move_right", hold=0.5)
            time.sleep(0.5)

    keyboard.send_event_on(pid, "rot_cw")
    time.sleep(0.03)
    keyboard.send_event_on(pid, "return")
    time.sleep(0.03)
    keyboard.send_event_off(pid, "return")
    keyboard.send_event_off(pid, "rot_cw")


def launch(limit_speed=False, music=False):
    process = subprocess.Popen([EMULATOR_PATH, ROM_PATH], stderr=None, stdout=None)
    print("Waiting for Start Screen")
    time.sleep(2)

    if limit_speed:
        print("Entering Speed Cheat")
        if enter_speed_cheat():
            print("Level 19 Speed Cheat Applied")
        else:
            print("Failed to apply cheat")
        time.sleep(3)
    else:
        print("Not applying speed cheat")

    start_template = cv2.imread(
        os.path.join(image_path, "push_start.png"), cv2.IMREAD_GRAYSCALE
    )
    for screen in capture.screenshot_generator(EMULATOR_NAME):
        if screen.shape[0] > 240:
            screen = cv2.resize(screen, (256, 240), interpolation=cv2.INTER_AREA)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(screen, start_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.4:
            break

    print("Starting the Game")
    keyboard.send_event(process.pid, "return", hold=0.5)
    time.sleep(1)

    if not music:
        music_off(process.pid)

    print("Select Game Type A")
    keyboard.send_event(process.pid, "return", hold=0.5)
    time.sleep(1)

    select_level(process.pid, 19)

    return process


def destroy(process):
    process.kill()
