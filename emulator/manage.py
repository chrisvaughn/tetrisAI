import os
import subprocess
import time

import cv2

from bot.detect import image_path
from emulator import capture, keyboard

EMULATOR_NAME = "Nestopia"
EMULATOR_PATH = "/Applications/Nestopia.app/Contents/MacOS/Nestopia"
ROM_PATH = "../ROMS/Tetris.nes"


def launch():
    process = subprocess.Popen([EMULATOR_PATH, ROM_PATH], stderr=None, stdout=None)
    print("Waiting for Start Screen")
    time.sleep(5)
    start_template = cv2.imread(
        os.path.join(image_path, "push_start.png"), cv2.IMREAD_GRAYSCALE
    )
    for screen in capture.screenshot_generator():
        if screen.shape[0] > 240:
            screen = cv2.resize(screen, (256, 240), interpolation=cv2.INTER_AREA)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(screen, start_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.4:
            break
    print("Starting the Game")
    keyboard.send_event(process.pid, "return")
    time.sleep(5)

    print("Selecting Level 9")
    seq = [
        "return",
        "return",
        "move_down",
        "move_right",
        "move_right",
        "move_right",
        "move_right",
        "return",
    ]
    for cmd in seq:
        keyboard.send_event(process.pid, cmd)
        time.sleep(1)

    return process


def destroy(process):
    process.kill()
