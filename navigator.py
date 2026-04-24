"""
Real-Time Visual Navigation System — YOLOv8 + OpenCV + Audio Guidance
GPU-accelerated (CUDA) for NVIDIA RTX 3050

Audio design
────────────
• Uses pyttsx3 (offline TTS) so there is zero network dependency and no
  perceptible startup delay.
• Speech runs on a dedicated daemon thread via AudioGuide so it never
  blocks the camera / inference loop.
• A cooldown timer prevents the same phrase from being repeated faster
  than AUDIO_COOLDOWN_SEC seconds, keeping output natural.
• Priority ordering: STOP > LEFT/RIGHT > CLEAR — a direction change
  always pre-empts a queued "Path is clear".
"""

import time
import threading
import queue

import cv2
import torch
import numpy as np
from ultralytics import YOLO


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
TRACKED_CLASSES = {"person", "chair", "bottle", "backpack", "couch", "dining table"}

PROXIMITY_THRESHOLD  = 160   # px – avoidance zone begins
CRITICAL_THRESHOLD   = 60    # px – STOP zone
CONFIDENCE_THRESHOLD = 0.40  # minimum detection confidence

ARROW_LENGTH     = 80
ARROW_THICKNESS  = 4
ARROW_TIP_RATIO  = 0.3

# Most built-in and USB webcams deliver a horizontally-mirrored feed.
# Flipping here (before inference) means every coordinate — including
# LEFT/RIGHT guidance and arrow direction — matches the real world.
# Set to False only if your camera already outputs a non-mirrored image.
MIRROR_WEBCAM = True

# Seconds before the same guidance phrase may be spoken again.
AUDIO_COOLDOWN_SEC = 2.5

# TTS voice speed (words per minute).  175 = natural, 160 = clear & slower.
TTS_RATE = 165

# Phrases spoken for each guidance token.
AUDIO_PHRASES: dict[str, str] = {
    "CLEAR": "Path is clear",
    "LEFT":  "Move left",
    "RIGHT": "Move right",
    "STOP":  "Stop! Obstacle ahead",
}

# Priority of each token (higher = more urgent, pre-empts lower).
AUDIO_PRIORITY: dict[str, int] = {
    "CLEAR": 0,
    "LEFT":  1,
    "RIGHT": 1,
    "STOP":  2,
}


# ──────────────────────────────────────────────
# Colour palette  (BGR)
# ──────────────────────────────────────────────
GREEN  = (0, 200, 60)
YELLOW = (0, 210, 255)
RED    = (0, 0, 220)
WHITE  = (255, 255, 255)
CYAN   = (220, 210, 0)
ORANGE = (0, 140, 255)


# ══════════════════════════════════════════════════════════════════════════════
# AudioGuide — non-blocking TTS on a background thread
# ══════════════════════════════════════════════════════════════════════════════

class AudioGuide:
    """
    Wraps pyttsx3 in a dedicated daemon thread so speech never blocks
    the main camera loop.

    Usage
    ─────
        audio = AudioGuide()
        audio.start()
        audio.speak("STOP")   # non-blocking; returns immediately
        audio.stop()          # call on shutdown

    Internal mechanics
    ──────────────────
    • A Queue of size 1 holds the *pending* phrase.  If the main thread posts
      a new, higher-priority message before the worker thread has picked up
      the previous one, the old message is discarded.
    • The worker calls engine.say() + engine.runAndWait() which is
      synchronous inside the worker thread – the pyttsx3-safe pattern.
    • A cooldown timestamp prevents the same token from being re-queued more
      often than AUDIO_COOLDOWN_SEC, avoiding chatter during sustained alerts.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._last_spoken_token: str = ""
        self._last_spoken_time: float = 0.0
        self._lock = threading.Lock()   # guards _last_spoken_* fields
        self._running = False

        # Probe TTS availability without holding a global engine reference.
        # Probe if espeak-ng is installed on the system
        import shutil
        if shutil.which("espeak-ng"):
            self._available = True
        else:
            print("[AudioGuide] espeak-ng not found – audio disabled.")
            self._available = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._available:
            self._running = True
            self._thread.start()
            print("[AudioGuide] TTS thread started.")

    def stop(self) -> None:
        self._running = False
        # Unblock the worker if it is waiting on the queue.
        try:
            self._queue.put_nowait("__QUIT__")
        except queue.Full:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, token: str) -> None:
        """
        Request speech for the given guidance token.

        Rules applied before queuing
        ────────────────────────────
        1. Token must be recognised in AUDIO_PHRASES.
        2. Cooldown: same token cannot be re-spoken within AUDIO_COOLDOWN_SEC.
        3. Priority: a lower-priority token will NOT displace a queued
           higher-priority token still waiting to be spoken.
        """
        if not self._available or not self._running:
            return
        if token not in AUDIO_PHRASES:
            return

        now = time.monotonic()
        with self._lock:
            same_token  = token == self._last_spoken_token
            within_cool = (now - self._last_spoken_time) < AUDIO_COOLDOWN_SEC
            if same_token and within_cool:
                return  # still cooling down for this token

            new_priority = AUDIO_PRIORITY.get(token, 0)

            # Drain any pending item and compare priorities.
            pending = None
            try:
                pending = self._queue.get_nowait()
            except queue.Empty:
                pass

            if pending is not None and pending != "__QUIT__":
                pending_priority = AUDIO_PRIORITY.get(pending, 0)
                if pending_priority > new_priority:
                    # Keep the higher-priority item; discard the new request.
                    self._queue.put_nowait(pending)
                    return
                # New token wins or ties; fall through to enqueue it.

            try:
                self._queue.put_nowait(token)
                self._last_spoken_token = token
                self._last_spoken_time  = now
            except queue.Full:
                pass  # Rare race — skip silently.

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _worker(self) -> None:
        """
        Runs in its own thread. Uses espeak-ng natively for Arch Linux.
        """
        import os # Imported locally for the system call
        
        while self._running:
            try:
                token = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if token == "__QUIT__":
                break

            phrase = AUDIO_PHRASES.get(token, "")
            if phrase:
                # Native Arch Linux call. 
                # -s sets the speed to your preferred TTS_RATE (165)
                os.system(f'espeak-ng -s {TTS_RATE} "{phrase}"')

# ══════════════════════════════════════════════════════════════════════════════
# Detection — data class
# ══════════════════════════════════════════════════════════════════════════════

class Detection:
    """Lightweight data holder for a single YOLO bounding-box result."""

    def __init__(self, label: str, conf: float, x1: int, y1: int, x2: int, y2: int):
        self.label = label
        self.conf  = conf
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

        # ── Centre coordinates ──────────────────────────────────────────────
        # cx = x1 + (x2 - x1) / 2  →  horizontal midpoint of the box
        # cy = y1 + (y2 - y1) / 2  →  vertical midpoint of the box
        self.cx = (x1 + x2) // 2
        self.cy = (y1 + y2) // 2

        self.w = x2 - x1
        self.h = y2 - y1


# ══════════════════════════════════════════════════════════════════════════════
# Navigator — main controller
# ══════════════════════════════════════════════════════════════════════════════

class Navigator:
    """
    Encapsulates model loading, inference, avoidance logic, rendering,
    and audio guidance.

    State kept here
    ───────────────
    • YOLO model instance
    • AudioGuide instance (owns its own TTS thread)
    • Frame dimensions (set on first frame)
    • Last computed guidance token
    """

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        # ── Device ───────────────────────────────────────────────────────────
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Navigator] Running on : {self.device.upper()}")
        if self.device == "cuda":
            print(f"[Navigator] GPU        : {torch.cuda.get_device_name(0)}")

        # ── YOLO ─────────────────────────────────────────────────────────────
        self.model = YOLO(model_path)
        self.model.to(self.device)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)   # warm-up

        # ── Frame state ───────────────────────────────────────────────────────
        self.frame_w: int = 0
        self.frame_h: int = 0
        self.guidance: str = "CLEAR"

        # ── Audio ─────────────────────────────────────────────────────────────
        self.audio = AudioGuide()
        self.audio.start()

    # ──────────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLOv8 inference; return filtered Detection objects."""
        results = self.model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
        detections: list[Detection] = []
        for box in results.boxes:
            label = self.model.names[int(box.cls)]
            if label not in TRACKED_CLASSES:
                continue
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(Detection(label, conf, x1, y1, x2, y2))
        return detections

    # ──────────────────────────────────────────────────────────────────────────
    # Avoidance logic
    # ──────────────────────────────────────────────────────────────────────────

    def compute_guidance(self, person: Detection, obstacle: Detection) -> str:
        """
        Decide the safest direction given one person and one obstacle.

        Coordinate math (after mirror flip)
        ─────────────────────────────────────
        The frame has already been flipped horizontally in run(), so pixel
        coordinates now match the real world:
            x = 0             →  user's physical LEFT
            x = frame_width   →  user's physical RIGHT

        Horizontal gap:
            dx = |person.cx - obstacle.cx|
            Small dx → person and obstacle share the same screen column
                     → potential collision.

        Clear-space heuristic:
            space_left  = obstacle.x1               (real-world pixels left of obstacle)
            space_right = frame_w - obstacle.x2     (real-world pixels right of obstacle)
            Steer toward whichever side has more room.

        Urgency levels:
            dx ≥ PROXIMITY_THRESHOLD  → CLEAR  (no action)
            CRITICAL ≤ dx < PROXIMITY → LEFT or RIGHT  (steer away)
            dx < CRITICAL_THRESHOLD   → STOP   (halt immediately)
        """
        dx = abs(person.cx - obstacle.cx)

        if dx >= PROXIMITY_THRESHOLD:
            return "CLEAR"

        if dx < CRITICAL_THRESHOLD:
            return "STOP"

        space_left  = obstacle.x1
        space_right = self.frame_w - obstacle.x2

        if person.cx < obstacle.cx:
            return "LEFT" if space_left >= space_right else "RIGHT"
        else:
            return "RIGHT" if space_right >= space_left else "LEFT"

    # ──────────────────────────────────────────────────────────────────────────
    # Rendering helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_box(self, frame: np.ndarray, det: Detection, colour: tuple) -> None:
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), colour, 2)
        label_text = f"{det.label} {det.conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame,
                      (det.x1, det.y1 - th - 8),
                      (det.x1 + tw + 4, det.y1),
                      colour, -1)
        cv2.putText(frame, label_text,
                    (det.x1 + 2, det.y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, (det.cx, det.cy), 5, WHITE,  -1)
        cv2.circle(frame, (det.cx, det.cy), 5, colour,  1)

    def _draw_guidance_arrow(self, frame: np.ndarray, guidance: str) -> None:
        """Arrow: ↑ green (CLEAR), ← yellow (LEFT), → yellow (RIGHT)."""
        ax = self.frame_w // 2
        ay = self.frame_h // 5
        L  = ARROW_LENGTH

        if guidance == "CLEAR":
            pt_start, pt_end = (ax, ay + L // 2), (ax, ay - L // 2)
            colour = GREEN
        elif guidance == "LEFT":
            pt_start, pt_end = (ax + L // 2, ay), (ax - L // 2, ay)
            colour = YELLOW
        elif guidance == "RIGHT":
            pt_start, pt_end = (ax - L // 2, ay), (ax + L // 2, ay)
            colour = YELLOW
        else:
            return

        cv2.arrowedLine(frame, pt_start, pt_end,
                        colour, ARROW_THICKNESS,
                        tipLength=ARROW_TIP_RATIO)

    def _draw_stop(self, frame: np.ndarray) -> None:
        text = "!! STOP !!"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 2.0, 3)
        tx = (self.frame_w - tw) // 2
        ty = self.frame_h // 4
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (tx - 10, ty - th - 10),
                      (tx + tw + 10, ty + 10),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, RED, 3, cv2.LINE_AA)

    def _draw_audio_indicator(self, frame: np.ndarray) -> None:
        """Top-right badge showing whether audio is active."""
        label  = "Audio: ON"  if self.audio._available else "Audio: OFF"
        colour = GREEN        if self.audio._available else (80, 80, 80)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, label,
                    (self.frame_w - tw - 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, fps: float) -> None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        badge = (f"GPU ({torch.cuda.get_device_name(0)[:12]})"
                 if self.device == "cuda" else "CPU")
        cv2.putText(frame, badge, (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1, cv2.LINE_AA)
        self._draw_audio_indicator(frame)

    # ──────────────────────────────────────────────────────────────────────────
    # Frame pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Per-frame pipeline:
            1. Detect objects
            2. Compute avoidance guidance
            3. Speak guidance (non-blocking)
            4. Render visual overlays
        """
        self.frame_h, self.frame_w = frame.shape[:2]

        detections = self.detect(frame)
        persons   = [d for d in detections if d.label == "person"]
        obstacles = [d for d in detections if d.label != "person"]

        for det in persons:
            self._draw_box(frame, det, GREEN)
        for det in obstacles:
            self._draw_box(frame, det, ORANGE)

        # ── Avoidance decision ───────────────────────────────────────────────
        self.guidance = "CLEAR"

        if persons and obstacles:
            person = persons[0]
            nearest_obstacle = min(obstacles, key=lambda o: abs(person.cx - o.cx))

            cv2.line(frame,
                     (person.cx, person.cy),
                     (nearest_obstacle.cx, nearest_obstacle.cy),
                     (180, 180, 180), 1, cv2.LINE_AA)

            self.guidance = self.compute_guidance(person, nearest_obstacle)

        # ── Audio (fire-and-forget) ──────────────────────────────────────────
        # AudioGuide.speak() returns immediately; the TTS thread handles timing.
        self.audio.speak(self.guidance)

        # ── Visual guidance ──────────────────────────────────────────────────
        if self.guidance == "STOP":
            self._draw_stop(frame)
        else:
            self._draw_guidance_arrow(frame, self.guidance)

        status_colour = {
            "CLEAR": GREEN, "LEFT": YELLOW,
            "RIGHT": YELLOW, "STOP": RED,
        }.get(self.guidance, WHITE)

        cv2.putText(frame, f"Guidance: {self.guidance}",
                    (10, self.frame_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_colour, 2, cv2.LINE_AA)

        self._draw_hud(frame, fps)
        return frame

    # ──────────────────────────────────────────────────────────────────────────
    # Run loop
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, camera_index: int = 0) -> None:
        """Open webcam and run the navigation + audio loop."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {camera_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("[Navigator] Press  Q  or  ESC  to quit.")

        prev_tick = cv2.getTickCount()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[Navigator] Frame capture failed – retrying …")
                    continue

                # ── Mirror correction ────────────────────────────────────────
                # cv2.flip(frame, 1) flips horizontally (flipCode=1).
                # After this, pixel x=0 is the user's physical LEFT and
                # x=frame_width is their physical RIGHT — matching reality.
                # All downstream coordinate math and arrow directions are
                # therefore correct without any further adjustment.
                if MIRROR_WEBCAM:
                    frame = cv2.flip(frame, 1)

                now = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (now - prev_tick)
                prev_tick = now

                frame = self.process_frame(frame, fps)
                cv2.imshow("Visual Navigation System", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.audio.stop()           # signal TTS thread to exit cleanly
            print("[Navigator] Exited cleanly.")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    nav = Navigator(model_path="yolov8n.pt")
    nav.run(camera_index=0)
