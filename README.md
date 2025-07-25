# Glitch & Morph Logo Animator

## Overview

This project is a high-performance Python script that creates complex, generative video animations from a single source image. It combines a multi-stage timeline of glitch and "damaged tape" effects with a continuous, random color-shifting engine and a customizable image overlay system.

The script is designed to be controlled by editing a clear, human-readable configuration section at the top of the `animate.py` file.

## How to Use

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Add Your Logo**: Place your main logo image (e.g., `pic.jpg`) in the project directory. Make sure the `IMAGE_PATH` variable at the top of `animate.py` matches the filename.

3.  **(Optional) Add Overlay Images**: Create a folder named `dinos`. Place any images you want to overlay on the animation inside this folder. For the best effect, use **PNG images with transparent backgrounds**. The script will automatically detect and use them.

4.  **Configure the Animation**: Open `animate.py` and modify the parameters in the configuration sections at the top of the file. See the reference guide below for what each setting does.

5.  **Run the Script**:
    ```bash
    python animate.py
    ```
    A new video file with a unique timestamped name (e.g., `video_combo_effect_1753406284.mp4`) will be created.

---

## Controlling the Animation: The "Prompt" System

Think of the configuration section in `animate.py` as a "prompt" that describes the animation you want. To change the video, you simply edit the values in this section.

### Prompt Reference: Available Effects & Parameters

#### Main Configuration
| Parameter | Description |
|---|---|
| `IMAGE_PATH` | The filename of your main logo/image. |
| `FPS` | Frames Per Second for the output video. `30` is standard. |
| `TOTAL_DURATION_S` | An *approximate* total duration for the video in seconds. The final length will vary slightly due to the random burst transitions. |

#### The Glitch Timeline
The animation is built on a sequence of 6 main sections. You can control the duration of each part.

| Parameter | Description |
|---|---|
| `SEC_0_FADE_IN_DURATION` | Duration for the initial fade-in of the base glitch effects. |
| `SEC_1_NORMAL_GLITCH_DURATION` | Duration of the standard, stable glitchy state. |
| `SEC_2_HEAVY_GLITCH_DURATION` | Duration for the more intense tape damage effects to be active. |
| `SEC_3_BW_DURATION` | Duration of the black and white phase. |
| `SEC_4_DESTRUCTION_DURATION` | Duration for the pixelation and fade-to-black sequence. |
| `SEC_5_RESTORATION_DURATION` | Duration for the final restoration of the clean logo. |

#### Effect Intensities
These parameters control the "strength" of individual visual effects.

| Parameter | Effect | Description |
|---|---|---|
| `NOISE_LEVEL` | **Industrial Noise** | Controls the amount of grainy, static-like noise. |
| `SCANLINE_INTENSITY` | **Scan Lines** | Controls the darkness of the horizontal CRT-style scan lines. |
| `WOBBLE_INTENSITY` | **Tape Jitter** | The vertical "wobble" of the image. |
| `CHROMATIC_ABERRATION` | **Color Fringing** | How far the Red and Blue color channels are shifted. |
| `GLITCH_INTENSITY` | **Block Glitches** | The number of horizontal "block" glitches that appear during the heavy damage section. |
| `HUM_BAR_INTENSITY` | **VCR Hum Bar** | The opacity of the dark, rolling bar seen during heavy damage. |

#### Color Morphing
This system continuously shifts the hue of your logo through a random sequence of colors.

| Parameter | Description |
|---|---|
| `COLOR_TRANSITION_S` | How many seconds it takes to smoothly morph from one color to the next. |
| `COLOR_PALETTE_BGR` | A list of target colors. You can add or remove any BGR color tuples to customize the palette. |

#### Image Overlays (The Dinosaurs)
This system overlays all images from a specified folder at set intervals.

| Parameter | Description |
|---|---|
| `DINO_DIR` | The name of the folder where you store your overlay images. |
| `DINO_INTERVAL_S` | A new overlay event will begin this many seconds after the previous one started. |
| `DINO_ONSCREEN_DURATION_S` | How long the overlay images remain on screen for each event. |
| `DINO_OPACITY` | The constant transparency of the overlay images. `0.5` is 50% opacity. |

#### Transitions
A randomized "pixel explosion" effect occurs between each major section of the glitch timeline.

| Parameter | Description |
|---|---|
| `BURST_DURATION_BASE` | The minimum duration for the explosion effect. |
| `BURST_DURATION_RANDOMNESS` | A random amount of time (0 to this value) is added to the base duration for variety. |

---

## Example Prompts

Here are some examples. To use one, simply copy the values into the configuration section of `animate.py`.

### Example 1: "Subtle & Moody"
A shorter, less intense animation with a focus on color shifting and minimal glitching.

```python
# --- Main Configuration ---
TOTAL_DURATION_S = 30

# --- Effect Intensities ---
NOISE_LEVEL = 0.1
SCANLINE_INTENSITY = 0.1
WOBBLE_INTENSITY = 2
CHROMATIC_ABERRATION = 3
GLITCH_INTENSITY = 1
HUM_BAR_INTENSITY = 0.2

# --- Glitch Effect Timeline (shorter) ---
SEC_0_FADE_IN_DURATION = 3
SEC_1_NORMAL_GLITCH_DURATION = 8
SEC_2_HEAVY_GLITCH_DURATION = 5
SEC_3_BW_DURATION = 3
SEC_4_DESTRUCTION_DURATION = 5
SEC_5_RESTORATION_DURATION = 5
```

### Example 2: "Fast & Chaotic"
A quick, aggressive animation with intense effects, frequent explosions, and fast color changes.

```python
# --- Main Configuration ---
TOTAL_DURATION_S = 20

# --- Effect Intensities ---
NOISE_LEVEL = 0.6
WOBBLE_INTENSITY = 10
GLITCH_INTENSITY = 15

# --- Color Morphing ---
COLOR_TRANSITION_S = 1.0 # Very fast color shifts

# --- Burst Effect ---
BURST_DURATION_BASE = 0.5
BURST_DURATION_RANDOMNESS = 0.5
```

### Example 3: "Dinosaur Invasion"
Focuses on making the dinosaur overlays the main event.

```python
# --- Dinosaur Overlay Configuration ---
DINO_INTERVAL_S = 4.0 # More frequent appearances
DINO_ONSCREEN_DURATION_S = 3.5
DINO_OPACITY = 0.75 # More opaque

# --- Glitch Effect Timeline (shorter) ---
SEC_0_FADE_IN_DURATION = 2
# ... etc.
``` 