import os
import subprocess
import time

import applescript
import cv2

from bot.detect import image_path
from emulator.capture import CaptureController
from emulator.keyboard import Keyboard

EMULATOR_NAME = "Nestopia"
EMULATOR_PATH = "/Applications/Nestopia.app/Contents/MacOS/Nestopia"
ROM_PATH = "../ROMS/Tetris.nes"

dir_path = os.path.dirname(os.path.realpath(__file__))


def enter_speed_cheat():
    r = applescript.run(os.path.join(dir_path, "enter_nestopia_cheat.scpt"))
    return r.code == 0


def enable_sound():
    r = applescript.run(os.path.join(dir_path, "enable_nestopia_sound.scpt"))
    return r.code == 0


def disable_sound():
    r = applescript.run(os.path.join(dir_path, "disable_nestopia_sound.scpt"))
    return r.code == 0


class Emulator:
    def __init__(self, limit_speed=True, music=False, level=19, sound=False):
        self.keyboard = None
        self.pid = None

        self.name = EMULATOR_NAME
        self.path = EMULATOR_PATH
        self.rom_path = ROM_PATH
        self.drop_keypress = None

        self.move_to_key_map = {
            "rot_ccw": "shift",
            "rot_cw": "option",
            "move_left": "left",
            "move_right": "right",
            "move_down": "down",
        }

        self.capturer = CaptureController(self.name, 50, False)
        self.process = self.launch(limit_speed, music, level, sound)

    def move_to_key(self, move):
        return self.move_to_key_map.get(move, None)

    def send_move(self, move):
        keycode = self.move_to_key(move)
        if keycode is None:
            return False
        self.keyboard.press_key(keycode)

    def send_moves(self, moves):
        for move in moves:
            self.send_move(move)

    def send_multiple_moves(self, moves):
        keys = [k for k in [self.move_to_key(move) for move in moves] if k is not None]
        self.keyboard.simultaneous_key_press(keys, wait_min_before_press=True)

    def drop_on(self):
        if not self.drop_keypress:
            kp = self.keyboard.key_down("down")
            self.drop_keypress = kp

    def drop_off(self):
        if self.drop_keypress:
            self.keyboard.keypress_up(self.drop_keypress)
            self.drop_keypress = None

    def music_off(self):
        print("Turning Music Off")
        self.keyboard.press_keys(("down", "down", "down"), extra_wait=0.5)

    def select_level(self, level: int):
        print(f"Selecting Level {level}")
        use_option = False
        if level > 9:
            use_option = True
            level -= 10

        if level > 4:
            self.keyboard.press_key("down", extra_wait=0.3)
            level -= 5

        self.keyboard.press_keys(["right"] * level, extra_wait=0.3)

        if use_option:
            self.keyboard.simultaneous_key_press(("option", "return"), extra_wait=0.3)
        else:
            self.keyboard.press_key("return", extra_wait=0.3)

    def launch(self, limit_speed=False, music=False, level=0, sound=False):
        process = subprocess.Popen([self.path, self.rom_path], stderr=None, stdout=None)
        self.pid = process.pid
        self.keyboard = Keyboard(self.pid)

        if limit_speed:
            print("Entering Speed Cheat")
            if enter_speed_cheat():
                print("Level 19 Speed Cheat Applied")
            else:
                print("Failed to apply cheat")
        else:
            print("Not applying speed cheat")

        if sound:
            print("Enabling Sound")
            enable_sound()
        else:
            print("Disabling sound")
            disable_sound()

        start_template = cv2.imread(
            os.path.join(image_path, "push_start.png"), cv2.IMREAD_GRAYSCALE
        )
        while True:
            screen = self.capturer.latest_image()
            res = cv2.matchTemplate(screen, start_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.4:
                break
            time.sleep(0.1)

        print("Starting the Game")
        self.keyboard.press_key("return", extra_wait=0.5)
        time.sleep(1)

        if sound and not music:
            self.music_off()
            time.sleep(1)

        print("Select Game Type A")
        self.keyboard.press_key("return", extra_wait=0.5)

        time.sleep(0.5)

        self.select_level(level)
        return process

    def destroy(self):
        self.capturer.stop()
        self.process.kill()
        self.keyboard.keylog.output()

    def get_latest_image(self):
        if self.capturer.has_new_image:
            return self.capturer.latest_image()
        else:
            return None
