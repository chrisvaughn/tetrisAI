import time

import cv2
import numpy as np
from Cocoa import NSApplicationActivateIgnoringOtherApps, NSRunningApplication
from mss import mss
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGNullWindowID,
    kCGWindowListOptionOnScreenOnly,
)


class Capture:
    # window offsets to ignore
    titlebar = 25
    buffer_y = 20

    def __init__(self, emulator_name, fps=30):
        self.fps = fps
        self.emulator_name = emulator_name
        self.images = self.screenshot_generator()
        self.capture_time = 1 / fps
        self.enabled = True
        self.location = None
        self.last_location_check = 0
        self.last_location_check_interval = 0.5

    @property
    def latest_image(self):
        return next(self.images)

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

    def screenshot_generator(self):
        with mss() as sct:
            while self.enabled:
                # time.sleep(self.capture_time)
                if (
                    time.time() - self.last_location_check
                    > self.last_location_check_interval
                ):
                    self.location = self._location(False)
                    self.last_location_check = time.time()
                if self.location is None:
                    print("Can't find emulator window")
                    continue

                mon = {
                    "top": self.location["y"] + Capture.titlebar,
                    "left": self.location["x"],
                    "width": self.location["width"],
                    "height": self.location["height"] - Capture.titlebar,
                }

                img = sct.grab(mon)
                img = np.array(img)
                if img.shape[0] > 240:
                    img = cv2.resize(img, (256, 240), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                yield img
