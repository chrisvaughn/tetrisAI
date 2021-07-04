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
    if move == "rot_cw":
        return 61
    if move == "move_left":
        return 123
    if move == "move_right":
        return 124
    if move == "move_down":
        return 125
    if move == "return":
        return 36
    if move == "noop":
        return None


def send_event(pid, move, hold=None):
    keycode = move_to_key(move)
    if keycode is None:
        return
    if hold is None:
        hold = 0.03
    event = CGEventCreateKeyboardEvent(src, keycode, True)
    CGEventPostToPid(pid, event)
    time.sleep(hold)
    event = CGEventCreateKeyboardEvent(src, keycode, False)
    CGEventPostToPid(pid, event)


def send_event_on(pid, move):
    keycode = move_to_key(move)
    if keycode is None:
        return
    event = CGEventCreateKeyboardEvent(src, keycode, True)
    CGEventPostToPid(pid, event)


def send_event_off(pid, move):
    keycode = move_to_key(move)
    if keycode is None:
        return
    event = CGEventCreateKeyboardEvent(src, keycode, False)
    CGEventPostToPid(pid, event)
