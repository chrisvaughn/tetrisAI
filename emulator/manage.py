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
    keyboard.send_event(process.pid, "return")
    time.sleep(1)
    keyboard.send_event(process.pid, "return")

    return process


def destroy(process):
    process.kill()
