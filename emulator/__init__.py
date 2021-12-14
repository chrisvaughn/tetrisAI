import os
import subprocess
import time

import applescript
import cv2
from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPostToPid,
    CGEventSourceCreate,
    kCGEventSourceStateHIDSystemState,
)

from bot.detect import image_path
from emulator import capture

EMULATOR_NAME = "Nestopia"
EMULATOR_PATH = "/Applications/Nestopia.app/Contents/MacOS/Nestopia"
ROM_PATH = "../ROMS/Tetris.nes"

dir_path = os.path.dirname(os.path.realpath(__file__))


def enter_speed_cheat():
    r = applescript.run(os.path.join(dir_path, "enter_nestopia_cheat.scpt"))
    return r.code == 0


class Keyboard:
    def __init__(self):
        self.key_map = {
            "rot_ccw": 60,
            "rot_cw": 61,
            "move_left": 123,
            "move_right": 124,
            "move_down": 125,
            "return": 36,
        }
        self.default_hold_time = 0.03
        self.min_hold_time = 0.001
        self.source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)

    def move_to_key(self, move):
        return self.key_map.get(move, None)

    def key_down(self, pid, move) -> bool:
        keycode = self.move_to_key(move)
        if keycode is None:
            return False
        event = CGEventCreateKeyboardEvent(self.source, keycode, True)
        CGEventPostToPid(pid, event)
        return True

    def key_up(self, pid, move) -> bool:
        keycode = self.move_to_key(move)
        if keycode is None:
            return False
        event = CGEventCreateKeyboardEvent(self.source, keycode, False)
        CGEventPostToPid(pid, event)
        return True


class Emulator:
    def __init__(self, limit_speed=True, music=False):
        self.name = EMULATOR_NAME
        self.path = EMULATOR_PATH
        self.rom_path = ROM_PATH
        self.pid = None
        self.keyboard = Keyboard()
        self.capture = capture.Capture(self.name)
        self.process = self.launch(limit_speed, music)

    def press_key(self, move, hold=None):
        if hold is None:
            hold = self.keyboard.default_hold_time
        if self.keyboard.key_down(self.pid, move):
            time.sleep(hold)
            self.keyboard.key_up(self.pid, move)

    def press_keys(self, moves, hold=None):
        if hold is None:
            hold = self.keyboard.default_hold_time
        for move in moves:
            if self.keyboard.key_down(self.pid, move):
                time.sleep(self.keyboard.min_hold_time)

        time.sleep(hold)

        for move in reversed(moves):
            if self.keyboard.key_up(self.pid, move):
                time.sleep(self.keyboard.min_hold_time)

    def drop_on(self):
        self.keyboard.key_down(self.pid, "move_down")

    def drop_off(self):
        self.keyboard.key_up(self.pid, "move_down")

    def music_off(self):
        print("Turning Music Off")
        for _ in range(3):
            self.press_key("move_down", hold=0.5)
            time.sleep(0.5)

    def select_level(self, level: int):
        print(f"Selecting Level {level}")
        if level < 5:
            for _ in range(level):
                self.press_key("move_right", hold=0.5)
                time.sleep(0.5)
        elif level < 10:
            self.press_key("move_down", hold=0.5)
            for _ in range(level - 5):
                self.press_key("move_right", hold=0.5)
                time.sleep(0.5)

        if level < 10:
            self.press_key("return", hold=0.5)
            return

        if level < 15:
            for _ in range(level - 10):
                self.press_key("move_right", hold=0.5)
                time.sleep(0.5)
        elif level < 20:
            self.press_key("move_down", hold=0.5)
            for _ in range(level - 15):
                self.press_key("move_right", hold=0.5)
                time.sleep(0.5)

        self.keyboard.key_down(self.pid, "rot_cw")
        time.sleep(0.03)
        self.keyboard.key_down(self.pid, "return")
        time.sleep(0.03)
        self.keyboard.key_up(self.pid, "return")
        self.keyboard.key_up(self.pid, "rot_cw")

    def launch(self, limit_speed=False, music=False):
        process = subprocess.Popen([self.path, self.rom_path], stderr=None, stdout=None)
        self.pid = process.pid
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
        for screen in self.capture.images:
            res = cv2.matchTemplate(screen, start_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.4:
                break

        print("Starting the Game")
        self.press_key("return", hold=0.5)
        time.sleep(1)

        if not music:
            self.music_off()

        print("Select Game Type A")
        self.press_key("return", hold=0.5)
        time.sleep(1)

        self.select_level(19)
        return process

    def destroy(self):
        self.capture.stop_capturing()
        self.process.kill()

    def get_latest_image(self):
        return self.capture.latest_image
