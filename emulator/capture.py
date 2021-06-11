import os

import numpy as np
from mss import mss

from Cocoa import NSRunningApplication, NSApplicationActivateIgnoringOtherApps
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGNullWindowID,
    kCGWindowListOptionOnScreenOnly,
)


dir_path = os.path.dirname(os.path.realpath(__file__))

# fceux window offsets to ignore
titlebar = 25
buffer_y = 20


def get_emulator_pid(name):
    wl = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    for v in wl:
        if name in v.valueForKey_("kCGWindowOwnerName"):
            return int(v.valueForKey_("kCGWindowOwnerPID"))


def get_emulator_location(name, bring_to_front=False):
    wl = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    for v in wl:
        if name in v.valueForKey_("kCGWindowOwnerName"):
            if v.valueForKey_("kCGWindowBounds") is None:
                return None
            if bring_to_front:
                pid = int(v.valueForKey_("kCGWindowOwnerPID"))
                x = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
                x.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            return {
                "x": int(v.valueForKey_("kCGWindowBounds").valueForKey_("X")),
                "y": int(v.valueForKey_("kCGWindowBounds").valueForKey_("Y")),
                "height": int(v.valueForKey_("kCGWindowBounds").valueForKey_("Height")),
                "width": int(v.valueForKey_("kCGWindowBounds").valueForKey_("Width")),
            }


def screenshot_generator():
    with mss() as sct:
        while True:
            location = get_emulator_location("Nestopia", False)
            if location is None:
                print("Can't find emulator window")
                return
            mon = {
                "top": location["y"] + titlebar,
                "left": location["x"],
                "width": location["width"],
                "height": location["height"] - titlebar,
            }

            img = sct.grab(mon)
            img = np.array(img)

            yield img
