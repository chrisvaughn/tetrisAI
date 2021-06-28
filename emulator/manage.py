import subprocess
import time

from emulator import keyboard

EMULATOR_NAME = "Nestopia"
EMULATOR_PATH = "/Applications/Nestopia.app/Contents/MacOS/Nestopia"
ROM_PATH = "../ROMS/Tetris.nes"


def launch():
    process = subprocess.Popen([EMULATOR_PATH, ROM_PATH], stderr=None, stdout=None)
    time.sleep(12)
    keyboard.send_event(process.pid, "return")
    time.sleep(1)

    seq = [
        "return",
        "return",
        "move_down",
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
