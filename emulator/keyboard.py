import csv
import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPostToPid,
    CGEventSourceCreate,
    kCGEventSourceStateHIDSystemState,
)

keys_to_keycodes = {
    "shift": 60,
    "option": 61,
    "left": 123,
    "right": 124,
    "down": 125,
    "return": 36,
}


@dataclass
class KeyPress:
    id: int
    keycode: int
    key: str
    pressed_at: float = time.time()
    released_at: float = None

    def as_list(self, timedelta):
        return [
            self.id,
            self.keycode,
            self.pressed_at - timedelta,
            self.released_at - timedelta,
        ]


class KeyboardDisplay:
    def __init__(self):
        self.key_states = {key: False for key in keys_to_keycodes.keys()}
        self.block_size = 100
        self.display()

    def press(self, key):
        self.key_states[key] = True
        self.display()

    def unpress(self, key):
        self.key_states[key] = False
        self.display()

    def display(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        color = (255, 255, 0)
        thickness = 1
        counter = 0
        key_display = np.zeros(
            (2 * self.block_size, self.block_size * len(self.key_states), 3),
            dtype=np.uint8,
        )
        for key, state in self.key_states.items():
            cv2.rectangle(
                key_display,
                (counter * self.block_size, 0 * self.block_size),
                ((counter + 1) * self.block_size, (0 + 1) * self.block_size),
                (255, 255, 255),
                cv2.FILLED if state else None,
            )
            org = (
                counter * self.block_size + 10,
                self.block_size + self.block_size // 2,
            )
            cv2.putText(
                key_display,
                key,
                org,
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            counter += 1
        cv2.imshow("Keyboard", key_display)
        cv2.waitKey(1)


class KeyLog:
    def __init__(self):
        self.keypresses: List[KeyPress] = []
        self.start_time = time.time()

    def capture(self, keypress: KeyPress):
        self.keypresses.append(keypress)

    def output(self):
        print("Saving keylog to keypress_log.csv")
        with open("keypress_log.csv", "w") as f:
            write = csv.writer(f)
            for keypress in self.keypresses:
                write.writerow(keypress.as_list(self.start_time))


class Keyboard:
    def __init__(self, pid: int, debug: bool = False, display: bool = False):
        self.emulator_detection_time = 0.02
        self.min_time_between_key_presses = self.emulator_detection_time * 10
        self.source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
        self.pid = pid
        self.last_keyup_time = 0.0
        self.debug = debug
        self.keylog = KeyLog()
        self.key_counter = 0
        if display:
            self.display = KeyboardDisplay()
            self.display.display()
        else:
            self.display = None

    @property
    def key_hold_time(self):
        return self.emulator_detection_time

    def key_down(self, key: str) -> KeyPress:
        keycode = keys_to_keycodes[key]
        event = CGEventCreateKeyboardEvent(self.source, keycode, True)
        pressed_at = time.time()
        CGEventPostToPid(self.pid, event)
        self.key_counter += 1
        if self.display:
            self.display.press(key)
        return KeyPress(
            id=self.key_counter, keycode=keycode, key=key, pressed_at=pressed_at
        )

    def _release(self, keycode: int) -> float:
        event = CGEventCreateKeyboardEvent(self.source, keycode, False)
        t = time.time()
        CGEventPostToPid(self.pid, event)
        self.last_keyup_time = t
        return t

    def keypress_up(self, keypress: KeyPress, extra_wait: float = 0.0):
        now = time.time()
        if now - keypress.pressed_at < self.key_hold_time + extra_wait:
            time.sleep(self.key_hold_time - (now - keypress.pressed_at) + extra_wait)
        release_at = self._release(keypress.keycode)
        keypress.released_at = release_at
        self.keylog.capture(keypress)
        if self.display:
            self.display.unpress(keypress.key)

    def press_key(
        self, key: str, wait_min_before_press: bool = False, extra_wait: float = 0.0
    ):
        if wait_min_before_press:
            wait_time = max(
                0.0,
                self.last_keyup_time + self.min_time_between_key_presses - time.time(),
            )
            if self.debug:
                print(f"Waiting {wait_time} seconds before pressing {key}")
            time.sleep(wait_time)
        k = self.key_down(key)
        self.keypress_up(k, extra_wait=extra_wait)

    def press_keys(self, keys: List[str], extra_wait: float = 0.0):
        for key in keys:
            self.press_key(key, wait_min_before_press=True, extra_wait=extra_wait)

    def simultaneous_key_press(
        self,
        keys: Tuple[str],
        wait_min_before_press: bool = False,
        extra_wait: float = 0.0,
    ):
        if wait_min_before_press:
            wait_time = max(
                0.0,
                self.last_keyup_time + self.min_time_between_key_presses - time.time(),
            )
            if self.debug:
                print(f"Waiting {wait_time} seconds before pressing {keys}")
            time.sleep(wait_time)
        keypresses = []
        for key in keys:
            keypresses.append(self.key_down(key))
            time.sleep(self.emulator_detection_time / 1.4)
        for keypress in reversed(keypresses):
            self.keypress_up(keypress, extra_wait)
