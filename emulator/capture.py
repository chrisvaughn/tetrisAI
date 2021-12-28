import time
from multiprocessing import Process, shared_memory

import cv2
import numpy as np
from Cocoa import NSApplicationActivateIgnoringOtherApps, NSRunningApplication
from mss import mss
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGNullWindowID,
    kCGWindowListOptionOnScreenOnly,
)

shape = (240, 256)
dtype = np.uint8


def capture_latest_image_into_shared_memory(fps, emulator_name, shared_mem):
    capturer = Capture(emulator_name, fps)
    capture_counter = 0
    # start_time = time.time()
    img = np.ndarray(shape, dtype=dtype, buffer=shared_mem.buf)
    while True:
        capture_start_time = time.time()
        capture_counter += 1
        # if capture_counter % 100 == 0:
        #     actual_fps = capture_counter / (time.time() - start_time)
        #     print(f"Capturing at {actual_fps:.2f} FPS")
        # copy the latest image into the shared memory
        try:
            img[:] = capturer.next_image[:]
        except TypeError:
            break
        # sleep for the remaining time to achieve the desired fps
        time.sleep(max(0, 1 / fps - (time.time() - capture_start_time)))


class CaptureController:
    def __init__(self, emulator_name, fps=50):
        self.shm = shared_memory.SharedMemory(
            create=True, size=shape[0] * shape[1] * dtype().itemsize
        )
        self.enabled = True
        self.process = Process(
            target=capture_latest_image_into_shared_memory,
            args=(
                fps,
                emulator_name,
                self.shm,
            ),
        )
        self.process.start()

    def stop(self):
        self.enabled = False
        self.process.terminate()
        self.shm.close()
        self.shm.unlink()

    def latest_image(self):
        return np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)


class Capture:
    # window offsets to ignore
    titlebar = 25
    buffer_y = 20

    def __init__(self, emulator_name, fps=50):
        self.fps = fps
        self.emulator_name = emulator_name
        self.capture_time = 1 / fps
        self.enabled = True
        self.location = None
        self.last_location_check = 0
        self.location_check_interval = 0.5  # seconds
        self.max_retries = 10

    @property
    def next_image(self):
        attempts = 0
        img = self.get_screenshot()
        while img is None and attempts < self.max_retries:
            time.sleep(0.1)
            img = self.get_screenshot()
            attempts += 1
        return img

    def stop_capturing(self):
        self.enabled = False

    def _location(self, bring_to_front=False):
        wl = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly, kCGNullWindowID
        )
        for v in wl:
            if self.emulator_name in v.valueForKey_("kCGWindowOwnerName"):
                if v.valueForKey_("kCGWindowBounds") is None:
                    return None
                if bring_to_front:
                    pid = int(v.valueForKey_("kCGWindowOwnerPID"))
                    x = NSRunningApplication.runningApplicationWithProcessIdentifier_(
                        pid
                    )
                    x.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
                return {
                    "x": int(v.valueForKey_("kCGWindowBounds").valueForKey_("X")),
                    "y": int(v.valueForKey_("kCGWindowBounds").valueForKey_("Y")),
                    "height": int(
                        v.valueForKey_("kCGWindowBounds").valueForKey_("Height")
                    ),
                    "width": int(
                        v.valueForKey_("kCGWindowBounds").valueForKey_("Width")
                    ),
                }

    def get_screenshot(self):
        with mss() as sct:
            if time.time() - self.last_location_check > self.location_check_interval:
                self.location = self._location(False)
                self.last_location_check = time.time()
            if self.location is None:
                print("Can't find emulator window")
                return None

            mon = {
                "top": self.location["y"] + Capture.titlebar,
                "left": self.location["x"],
                "width": self.location["width"],
                "height": self.location["height"] - Capture.titlebar,
            }

            img = sct.grab(mon)
            img = np.array(img)
            if img.shape[0] < 240:
                return None

            if img.shape[0] > 240:
                img = cv2.resize(img, (256, 240), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
