import os
import random
import subprocess
import time

import cv2
import numpy as np

from emulator.capture import CaptureController

FCEUX_NAME = "fceux"
FCEUX_PATH = "fceux"  # must be on PATH, or set to absolute path
ROM_PATH = "../ROMS/Tetris.nes"

CMD_FILE = "/tmp/fceux_tetris_cmd"
ACK_FILE = "/tmp/fceux_tetris_ack"

HOLD_FRAMES = 1
GAP_FRAMES = 1
ACK_TIMEOUT = 5.0

dir_path = os.path.dirname(os.path.realpath(__file__))


class FCEUXEmulator:
    def __init__(self, music=False, level=19, sound=False, manual_start=False):
        self.name = FCEUX_NAME
        self._dropping = False

        # NES button names used by joypad.set() in the Lua bridge
        self._move_to_button = {
            "rot_ccw": "B",
            "rot_cw": "A",
            "move_left": "left",
            "move_right": "right",
            "move_down": "down",
        }

        for f in (CMD_FILE, ACK_FILE):
            if os.path.exists(f):
                os.remove(f)

        lua_script = os.path.join(dir_path, "fceux_bridge.lua")

        self.capturer = CaptureController(FCEUX_NAME, 60, False)
        self.process = self._launch(lua_script, music, level, sound, manual_start)

    # ------------------------------------------------------------------
    # IPC
    # ------------------------------------------------------------------

    def _send_command(self, cmd: str, timeout: float = ACK_TIMEOUT) -> bool:
        with open(CMD_FILE, "w") as f:
            f.write(cmd + "\n")

        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.exists(ACK_FILE):
                os.remove(ACK_FILE)
                return True
            time.sleep(0.001)

        # Timed out -- the bridge may be stuck (e.g. a game-over/topout
        # animation froze its frame loop) and never wrote an ack. Clear any
        # stale command/ack files so the next command starts clean, and let
        # the caller carry on (e.g. the main loop's game_over() check) instead
        # of crashing the whole run.
        print(f"  [fceux] WARNING: bridge ack timeout for command: {cmd}")
        for f in (CMD_FILE, ACK_FILE):
            if os.path.exists(f):
                os.remove(f)
        return False

    def _press(self, buttons: list[str], hold_frames=HOLD_FRAMES, gap_frames=GAP_FRAMES):
        self._send_command(f"press:{','.join(buttons)}:{hold_frames}:{gap_frames}")

    def ping(self, timeout: float = ACK_TIMEOUT) -> bool:
        """Return True if the Lua bridge is alive and responding."""
        return self._send_command("ping", timeout=timeout)

    # ------------------------------------------------------------------
    # Public interface (mirrors emulator/__init__.py Emulator)
    # ------------------------------------------------------------------

    def send_move(self, move: str):
        button = self._move_to_button.get(move)
        if button:
            self._press([button])

    def send_moves(self, moves: list[str]):
        for move in moves:
            self.send_move(move)

    def send_multiple_moves(self, moves: list[str]):
        buttons = [self._move_to_button[m] for m in moves if m in self._move_to_button]
        if buttons:
            self._press(buttons)

    def send_move_sequence(self, move_sequence: list[tuple], hold_frames=HOLD_FRAMES, gap_frames=GAP_FRAMES):
        """Send a whole piece's keypress sequence as a single blocking IPC
        round trip, instead of one round trip per keypress. This avoids
        accumulating per-call IPC overhead while a piece is in flight, which
        otherwise leaves the next piece falling uncontrolled long enough to
        land on the stack before we can plan/send its moves."""
        steps = []
        for moves in move_sequence:
            buttons = [self._move_to_button[m] for m in moves if m in self._move_to_button]
            if buttons:
                steps.append(",".join(buttons))
        if not steps:
            return
        self._send_command(f"press_seq:{'|'.join(steps)}:{hold_frames}:{gap_frames}")

    def drop_on(self):
        if not self._dropping:
            self._send_command("hold:down")
            self._dropping = True

    def drop_off(self):
        if self._dropping:
            self._send_command("release:down")
            self._dropping = False

    def get_latest_image(self):
        if self.capturer.has_new_image():
            return self.capturer.latest_image()
        return None

    def destroy(self):
        self.capturer.stop()
        self.process.kill()
        for f in (CMD_FILE, ACK_FILE):
            if os.path.exists(f):
                os.remove(f)

    # ------------------------------------------------------------------
    # Menu navigation helpers
    # ------------------------------------------------------------------

    def _current_screen(self) -> np.ndarray:
        return self.capturer.latest_image()

    def _load_template(self, name: str) -> np.ndarray | None:
        path = os.path.join(dir_path, "templates", f"{name}.png")
        if not os.path.exists(path):
            return None
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def _wait_for_template(self, template_name: str, timeout: float = 15.0, threshold: float = 0.6) -> bool:
        """Block until the named template is visible on screen.

        Returns True when matched, False on timeout.
        Falls back to a fixed sleep when the template file does not exist.
        """
        tmpl = self._load_template(template_name)
        if tmpl is None:
            print(f"  [{template_name}] template not found — sleeping 2s as fallback")
            time.sleep(2.0)
            return True

        print(f"  waiting for '{template_name}' screen…")
        deadline = time.time() + timeout
        last_report = time.time()
        best = 0.0
        while time.time() < deadline:
            screen = self.capturer.latest_image()
            if screen is None:
                time.sleep(0.05)
                continue
            res = cv2.matchTemplate(screen, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            best = max(best, max_val)
            if max_val >= threshold:
                print(f"  '{template_name}' detected (score={max_val:.2f})")
                return True
            if time.time() - last_report >= 2.0:
                print(f"  [{template_name}] still waiting… best={best:.2f} current={max_val:.2f}")
                last_report = time.time()
            time.sleep(0.05)

        print(f"  Warning: '{template_name}' not detected within {timeout}s (best score={best:.2f})")
        return False

    def _nav_press(self, button: str, extra_wait: float = 0.0):
        self._press([button], hold_frames=3, gap_frames=2)
        if extra_wait:
            time.sleep(extra_wait)

    def _nav_press_multi(self, buttons: list[str], extra_wait: float = 0.0):
        self._press(buttons, hold_frames=3, gap_frames=2)
        if extra_wait:
            time.sleep(extra_wait)

    def _music_off(self):
        print("Turning Music Off")
        for _ in range(3):
            self._nav_press("down", extra_wait=0.5)

    def _select_level(self, level: int):
        print(f"Selecting Level {level}")
        use_a = level > 9
        if use_a:
            level -= 10

        if level > 4:
            print("  cursor: down")
            self._nav_press("down", extra_wait=0.4)
            level -= 5

        for i in range(level):
            print(f"  cursor: right ({i + 1}/{level})")
            self._nav_press("right", extra_wait=0.4)

        if use_a:
            print("  confirm: hold-A + start (level +10)")
            # Hold A first, then press START while A is still held, then release A.
            # Simultaneous release caused NES Tetris to ignore the input.
            self._send_command("hold:A")
            time.sleep(0.1)
            self._nav_press("start")
            time.sleep(0.1)
            self._send_command("release:A")
        else:
            print("  confirm: start")
            self._nav_press("start")

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def _launch(self, lua_script: str, music: bool, level: int, sound: bool, manual_start: bool):
        lua_script = os.path.abspath(lua_script)
        # ROM_PATH is relative to cwd (project root), same as Nestopia
        process = subprocess.Popen(
            [FCEUX_PATH, "--loadlua", lua_script, ROM_PATH],
            stderr=None,
            stdout=None,
        )

        if manual_start:
            print("FCEUX launched. Start the game manually, then press Enter here to begin the bot.")
            input()
            return process

        # Brief wait for FCEUX window to appear before template polling starts
        time.sleep(1)

        self._wait_for_template("title_screen")

        # Title screen → game-type / music select
        print("Starting the Game")
        self._nav_press("start")
        self._wait_for_template("game_type_select")

        if sound and not music:
            self._music_off()

        # Game-type select → (optional music select) → level select
        print("Select Game Type A")
        self._nav_press("start")
        # If a separate music-select screen exists in this ROM, navigate past it too
        if not self._wait_for_template("level_select", timeout=4.0):
            print("  level_select not seen — trying one more start press (music screen?)")
            self._nav_press("start")
            self._wait_for_template("level_select")

        # Random delay on the level-select screen to advance NES RNG.
        # Capped at 0.5s to keep bridge timeouts from occurring.
        rng_delay = random.uniform(0.0, 0.5)
        print(f"  RNG delay: {rng_delay:.2f}s")
        time.sleep(rng_delay)

        self._select_level(level)

        # Wait for the game to start
        self._wait_for_template("gameplay", timeout=15.0)

        print("Checking bridge is alive…")
        if self.ping(timeout=5.0):
            print("  bridge OK")
        else:
            print("  WARNING: bridge not responding after gameplay start")

        return process
