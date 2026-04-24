<div align="center">

# 🧭 Visual Navigation System

**Real-time obstacle detection and avoidance with visual + spoken guidance**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00CFFF?style=flat)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat)](LICENSE)

*Detects people and common obstacles from your webcam, computes a safe escape direction in real time, and tells you where to go — through on-screen arrows **and** spoken audio.*

</div>

---

## ✨ Features

- **GPU-accelerated inference** — YOLOv8 nano running on CUDA (NVIDIA RTX 3050 and up)
- **Dual guidance output** — coloured on-screen arrows *and* offline text-to-speech, simultaneously
- **Mirror-corrected** — webcam feed is horizontally flipped before inference so LEFT/RIGHT match the real world
- **Non-blocking audio** — TTS runs on its own daemon thread; the camera loop is never paused waiting for speech
- **Smart audio queue** — cooldown timer + priority system (STOP beats LEFT/RIGHT beats CLEAR) keeps announcements natural and urgent
- **Clean OOP design** — `Detection`, `AudioGuide`, and `Navigator` classes with clear separation of concerns
- **Zero cloud dependency** — pyttsx3 uses the OS speech engine (SAPI5 / espeak / nsss); no API keys, no network calls

---

## 🎬 How It Works

```
┌─────────────────────────────────────────────────────────┐
│                      Webcam Frame                        │
└───────────────────────────┬─────────────────────────────┘
                            │ cv2.flip(frame, 1)  ← mirror fix
                            ▼
┌─────────────────────────────────────────────────────────┐
│              YOLOv8n Inference  (GPU)                    │
│   Detects: person · chair · bottle · backpack ·         │
│            couch · dining table                          │
└───────────────────────────┬─────────────────────────────┘
                            │ bounding boxes + labels
                            ▼
┌─────────────────────────────────────────────────────────┐
│             Centre-Coordinate Extraction                  │
│   cx = (x1 + x2) / 2     cy = (y1 + y2) / 2            │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Proximity Check                         │
│   dx = │person.cx − obstacle.cx│                         │
│                                                          │
│   dx ≥ 160 px  ──────────────────►  CLEAR               │
│   60 ≤ dx < 160  ────────────────►  LEFT  or  RIGHT      │
│   dx < 60 px   ──────────────────►  STOP                 │
└───────────────────────────┬─────────────────────────────┘
                            │
                    ┌───────┴────────┐
                    ▼                ▼
        ┌─────────────────┐  ┌──────────────────┐
        │  Visual Overlay │  │   Audio Thread    │
        │  (OpenCV)       │  │   (pyttsx3)       │
        │  ↑ ← → STOP     │  │  "Move left" …   │
        └─────────────────┘  └──────────────────┘
```

### Clear-space decision (LEFT vs RIGHT)

When the system decides avoidance is needed, it picks the side with the most open room:

```
space_left  = obstacle.x1               ← pixels between obstacle and left wall
space_right = frame_width − obstacle.x2 ← pixels between obstacle and right wall
```

The person is then steered toward whichever side has more space, biased by which side they are already on so they are always pushed *away* from the obstacle rather than through it.

---

## 📐 Visual Overlays

| Overlay | Colour | Condition | Meaning |
|---|---|---|---|
| **↑ Arrow** | 🟢 Green | `dx ≥ 160 px` | Path is clear, continue forward |
| **← Arrow** | 🟡 Yellow | `60 ≤ dx < 160` | Move to the left |
| **→ Arrow** | 🟡 Yellow | `60 ≤ dx < 160` | Move to the right |
| **!! STOP !!** | 🔴 Red | `dx < 60 px` | Critically close — halt immediately |
| Bounding box | 🟢 Green | — | Detected person |
| Bounding box | 🟠 Orange | — | Detected obstacle |
| Grey line | ⬜ Grey | — | Gap being measured between person and obstacle |
| `Audio: ON/OFF` | Top-right | — | TTS availability badge |
| `FPS` + device | Top-left | — | Performance HUD |

---

## 🔊 Audio Guidance

| State | Spoken phrase |
|---|---|
| Clear | *"Path is clear"* |
| Turn left | *"Move left"* |
| Turn right | *"Move right"* |
| Critical | *"Stop! Obstacle ahead"* |

### Audio system design

The `AudioGuide` class runs pyttsx3 on a **dedicated daemon thread** so speech never stalls the camera loop. Three rules govern every announcement:

1. **Cooldown (2.5 s)** — the same phrase cannot repeat until the cooldown expires, preventing robotic repetition during a sustained alert.
2. **Priority** — `STOP` (priority 2) always pre-empts a queued `LEFT`/`RIGHT` (priority 1), which pre-empts `CLEAR` (priority 0). You always hear the most urgent message.
3. **Queue size = 1** — stale guidance is discarded immediately; there is no backlog of outdated directions.

---

## 📁 Project Structure

```
visual_navigation/
│
├── navigator.py          ← Main application — run this
├── requirements.txt      ← pip dependencies
├── README.md             ← This file
│
└── yolov8n.pt            ← Auto-downloaded on first run (~6 MB)
```

---

## ⚙️ Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime (3.11 recommended) |
| PyTorch + CUDA | 2.x + CUDA 12.x | GPU inference |
| Ultralytics | 8.2+ | YOLOv8 model |
| OpenCV | 4.9+ | Video capture and rendering |
| NumPy | 1.26+ | Array operations |
| pyttsx3 | 2.90+ | Offline text-to-speech |

**System audio engine (required by pyttsx3):**

| OS | Engine | Extra step needed |
|---|---|---|
| Windows | SAPI5 | None — built-in |
| Linux | espeak | `sudo apt install espeak` |
| macOS | nsss | None — built-in |

---

## 🚀 Installation & Setup

### 1. Get the project

```bash
git clone https://github.com/your-username/visual-navigation.git
cd visual-navigation
```

Or just drop `navigator.py` and `requirements.txt` into a folder and open a terminal there.

---

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install PyTorch with CUDA

> ⚠️ **Do this before step 4.** Installing requirements first may pull in the CPU-only PyTorch build.

Visit **[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)** and choose your OS and CUDA version. For CUDA 12.1 the command is:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify the install worked:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected output:
# True  NVIDIA GeForce RTX 3050 Laptop GPU
```

---

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

**Linux only** — install the espeak TTS engine:

```bash
sudo apt install espeak
```

---

### 5. Run

```bash
python navigator.py
```

`yolov8n.pt` (~6 MB) is downloaded automatically on the very first run and cached locally. All subsequent runs skip this.

Press **`Q`** or **`ESC`** to exit cleanly.

---

## 🛠️ Configuration

All tunable constants live at the top of `navigator.py`. No changes elsewhere are needed.

| Constant | Default | Description |
|---|---|---|
| `TRACKED_CLASSES` | `person, chair, bottle, backpack, couch, dining table` | YOLO classes the system reacts to |
| `PROXIMITY_THRESHOLD` | `160 px` | Horizontal gap at which avoidance guidance begins |
| `CRITICAL_THRESHOLD` | `60 px` | Horizontal gap at which STOP is triggered |
| `CONFIDENCE_THRESHOLD` | `0.40` | Minimum YOLO confidence to accept a detection |
| `MIRROR_WEBCAM` | `True` | Flip the feed horizontally before inference (set `False` for non-mirroring cameras) |
| `AUDIO_COOLDOWN_SEC` | `2.5` | Minimum seconds between repeated announcements of the same phrase |
| `TTS_RATE` | `165` | Speech speed in words per minute |
| `AUDIO_PHRASES` | dict | Customise the exact words spoken for each guidance token |
| `ARROW_LENGTH` | `80 px` | Length of the guidance arrow drawn on screen |

### Changing camera index

If you have multiple cameras or your webcam is not at index 0, edit the last line of `navigator.py`:

```python
nav.run(camera_index=1)   # change 1 to your camera's index
```

---

## 🐛 Troubleshooting

**`RuntimeError: Cannot open camera at index 0`**
The webcam is not found or is already in use by another application. Try a different index (`camera_index=1`, `2`, …) or close any apps using the camera.

**`Audio: OFF` shown in the top-right / no speech heard**
The pyttsx3 engine could not initialise. On Linux, run `sudo apt install espeak`. Check that your system volume is not muted and the correct output device is selected.

**CUDA not detected — running on CPU**
Re-run the PyTorch CUDA install command (Step 3) making sure the `--index-url` flag is included. Verify your driver with `nvidia-smi`; it should show driver ≥ 525 and the correct GPU.

**Speech is repetitive or too frequent**
Increase `AUDIO_COOLDOWN_SEC` (e.g. to `4.0`) so announcements are spaced further apart.

**Speech is hard to understand**
Lower `TTS_RATE` to around `140`–`150` for slower, clearer pronunciation.

**Low FPS / laggy video**
Reduce the capture resolution inside `run()`:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```
Also ensure no other GPU-intensive processes are running in the background.

**Arrows pointing the wrong way**
If your camera does *not* mirror the image (e.g. an IP camera or virtual camera), set `MIRROR_WEBCAM = False` at the top of `navigator.py`.

---

## 📄 License

This project is released under the **MIT License** — free to use, modify, and distribute for any purpose.

---

<div align="center">
Built with <a href="https://ultralytics.com/">Ultralytics YOLOv8</a> · <a href="https://opencv.org/">OpenCV</a> · <a href="https://pyttsx3.readthedocs.io/">pyttsx3</a>
</div>
