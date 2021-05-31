import time

from Quartz import (
    CGEventSourceCreate,
    kCGEventSourceStateHIDSystemState,
    CGEventCreateKeyboardEvent,
    CGEventPostToPid,
)

src = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)


def move_to_key(move):
    if move == "rot_ccw":
        return 60
    if move == "move_left":
        return 123
    if move == "move_right":
        return 124
    if move == "move_down":
        return 125
    if move == "noop":
        return None


def send_event(pid, move):
    keycode = move_to_key(move)
    if keycode is None:
        return
    print("Sending Event")
    event = CGEventCreateKeyboardEvent(src, keycode, True)
    CGEventPostToPid(pid, event)
    time.sleep(0.03)
    event = CGEventCreateKeyboardEvent(src, keycode, False)
    CGEventPostToPid(pid, event)
