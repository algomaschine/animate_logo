import cv2
import numpy as np
import moviepy.editor as mp
import time
from tqdm import tqdm
import multiprocessing
import os
from multiprocessing import shared_memory


# remove background for dynos https://www.adobe.com/express/feature/image/remove-background

# --- Main Configuration ---
IMAGE_PATH = "pic.jpg" # Changed to .jpg
FPS = 30
TOTAL_DURATION_S = 60 # Approximate total duration

# --- Dinosaur Overlay Configuration ---
DINO_DIR = "dinos" # Folder to store your transparent dinosaur PNGs
DINO_INTERVAL_S = 5.0 # An appearance will start every 5 seconds
DINO_ONSCREEN_DURATION_S = 3.0 # Each appearance will last for 3 seconds
DINO_OPACITY = 0.5 # The constant opacity of the dinosaurs (0.0 to 1.0)

# --- Glitch Effect Timeline (Durations in Seconds) ---
SEC_0_FADE_IN_DURATION = 5
SEC_1_NORMAL_GLITCH_DURATION = 10
SEC_2_HEAVY_GLITCH_DURATION = 10
SEC_3_BW_DURATION = 8
SEC_4_DESTRUCTION_DURATION = 12
SEC_5_RESTORATION_DURATION = 10

# --- Color Morphing Configuration ---
COLOR_TRANSITION_S = 3 # Time to morph from one color to the next
COLOR_PALETTE_BGR = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
    (255, 255, 0), (255, 0, 0), (255, 0, 127), (127, 0, 255)
]

# --- Burst Effect Configuration ---
BURST_DURATION_BASE = 0.2
BURST_DURATION_RANDOMNESS = 0.3

# --- Effect-Specific Parameters ---
# You can tweak these values to change the look of the effects
NOISE_LEVEL = 0.4
SCANLINE_INTENSITY = 0.2
WOBBLE_INTENSITY = 5
CHROMATIC_ABERRATION = 5
GLITCH_INTENSITY = 5
HUM_BAR_INTENSITY = 0.4

# --- Timeline & Color Generation (Do not change) ---
rng = np.random.default_rng(seed=int(time.time()))

# Color morph timeline
def bgr_to_hue(c): return cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2HSV)[0][0][0]
HUE_PALETTE = [bgr_to_hue(c) for c in COLOR_PALETTE_BGR]
NUM_TRANSITIONS = int(TOTAL_DURATION_S / COLOR_TRANSITION_S)
HUE_SEQUENCE = [rng.choice(HUE_PALETTE)]
for _ in range(NUM_TRANSITIONS):
    possible_next = [h for h in HUE_PALETTE if h != HUE_SEQUENCE[-1]]
    HUE_SEQUENCE.append(rng.choice(possible_next))

# Glitch effect timeline
def secs(s): return int(s * FPS)
section_durations_s = [
    SEC_0_FADE_IN_DURATION, SEC_1_NORMAL_GLITCH_DURATION, SEC_2_HEAVY_GLITCH_DURATION,
    SEC_3_BW_DURATION, SEC_4_DESTRUCTION_DURATION, SEC_5_RESTORATION_DURATION
]
burst_durations_s = [BURST_DURATION_BASE + rng.random() * BURST_DURATION_RANDOMNESS for _ in range(len(section_durations_s))]
burst_seeds = [rng.integers(0, 10000) for _ in range(len(section_durations_s))]

timeline = []
current_frame = 0
for i, sec_duration in enumerate(section_durations_s):
    start, end = current_frame, current_frame + secs(sec_duration)
    timeline.append({'type': 'section', 'index': i, 'start': start, 'end': end})
    current_frame = end
    start, end = current_frame, current_frame + secs(burst_durations_s[i])
    timeline.append({'type': 'burst', 'index': i, 'start': start, 'end': end, 'seed': burst_seeds[i]})
    current_frame = end
FRAME_COUNT = current_frame

# --- Globals for Worker Processes ---
# These are set in each worker by the initializer
base_image_bgr, base_image_hsv = None, None
dino_images, dino_timeline = [], []
width, height = None, None
# For shared memory
shared_mem_name, result_shape, result_dtype = None, None, None

# --- Helper and Effect Functions ---
def init_worker(img_data, img_shape, dino_data, dino_timeline_data, shm_name, res_shape, res_dtype):
    """Initializes each worker process with necessary data and shared memory info."""
    global base_image_bgr, base_image_hsv, width, height, dino_images, dino_timeline
    global shared_mem_name, result_shape, result_dtype
    
    # Load base image
    base_image_bgr = np.frombuffer(img_data, dtype=np.uint8).reshape(img_shape)
    base_image_hsv = cv2.cvtColor(base_image_bgr, cv2.COLOR_BGR2HSV)
    height, width, _ = base_image_bgr.shape
    
    # Load dinos and timeline
    for d_data, d_shape in dino_data:
        dino_images.append(np.frombuffer(d_data, dtype=np.uint8).reshape(d_shape))
    dino_timeline = dino_timeline_data
    
    # Connect to shared memory
    shared_mem_name = shm_name
    result_shape = res_shape
    result_dtype = res_dtype


def lerp(v1, v2, p): return v1 * (1 - p) + v2 * p
def lerp_hue(v1, v2, p):
    diff = v2 - v1
    if abs(diff) > 90: diff -= 180 if diff > 0 else -180
    return (v1 + diff * p) % 180

def get_color_shifted_image(frame_index):
    """Calculates the color shift for the frame and returns a BGR image."""
    progress = (frame_index / FRAME_COUNT) * NUM_TRANSITIONS
    transition_idx = int(progress)
    progress_in_transition = progress - transition_idx
    
    start_hue = HUE_SEQUENCE[transition_idx]
    end_hue = HUE_SEQUENCE[transition_idx + 1]
    
    current_target_hue = lerp_hue(start_hue, end_hue, progress_in_transition)
    original_avg_hue = np.mean(base_image_hsv[:, :, 0])
    hue_shift = current_target_hue - original_avg_hue
    
    shifted_hsv = base_image_hsv.copy()
    shifted_hsv[:, :, 0] = (base_image_hsv[:, :, 0] + hue_shift) % 180
    return cv2.cvtColor(shifted_hsv, cv2.COLOR_HSV2BGR)

def apply_base_effects(frame, p, f_idx):
    aberration = int(lerp(0, CHROMATIC_ABERRATION, p))
    r = np.roll(frame[:, :, 2], -aberration, axis=1); r[:, -aberration:] = 0
    b = np.roll(frame[:, :, 0], aberration, axis=1); b[:, :aberration] = 0
    proc_frame = np.stack([b, frame[:, :, 1], r], axis=2)
    wobble = int(lerp(0, WOBBLE_INTENSITY, p) * np.sin(f_idx * np.pi / FPS))
    proc_frame = np.roll(proc_frame, wobble, axis=0)
    noise = np.random.normal(0, lerp(0, NOISE_LEVEL, p), (height, width, 3)) * 255
    proc_frame = np.clip(proc_frame + noise, 0, 255)
    scan_frame = proc_frame.astype(np.uint8)
    scan_intensity = lerp(0, SCANLINE_INTENSITY, p)
    for y in range(0, height, 4):
        scan_frame[y, :] = (scan_frame[y, :] * (1 - scan_intensity)).astype(np.uint8)
    return scan_frame

def apply_heavy_glitch(frame, p, f_idx):
    bar_h = height // 4
    bar_y = int((f_idx*15)%(height+bar_h)) - bar_h
    overlay = frame.copy(); cv2.rectangle(overlay,(0,bar_y),(width,bar_y+bar_h),(0,0,0),-1)
    frame = cv2.addWeighted(frame,1,overlay,lerp(0,HUM_BAR_INTENSITY,p),0)
    for _ in range(int(lerp(0,GLITCH_INTENSITY,p))):
        y,h,off = rng.integers(0,height-10), rng.integers(5,15), rng.integers(-50,50)
        frame[y:y+h,:] = np.roll(frame[y:y+h,:], off, axis=1)
    return frame

def apply_explosion(frame, p, seed):
    rng_exp = np.random.default_rng(seed)
    h,w,_=frame.shape; rad=rng_exp.integers(w//8,w//4); cx,cy=rng_exp.integers(rad,w-rad),rng_exp.integers(rad,h-rad)
    max_disp = rad*1.5; Y,X=np.ogrid[:h,:w]; dist=np.sqrt((X-cx)**2+(Y-cy)**2)
    mask = dist<=rad; out_frame=frame.copy(); vx,vy=X-cx,Y-cy; norm=np.sqrt(vx**2+vy**2); norm[norm==0]=1
    disp = (max_disp*p).astype(int); dx=(X+(vx/norm)*disp).clip(0,w-1).astype(int); dy=(Y+(vy/norm)*disp).clip(0,h-1).astype(int)
    out_frame[dy[mask],dx[mask]] = frame[mask]
    return out_frame

def apply_dino_overlay(base_frame, seed, frame_index):
    """
    EFFECT: Overlays TWO randomly selected dinosaurs at random positions.
    The positions are re-randomized on each frame.
    Handles both transparent PNGs and solid-background JPGs.
    """
    if not dino_images: return base_frame
    
    # By including the frame_index in the seed, we get new random
    # positions for the dinosaurs on every single frame.
    rng_dino = np.random.default_rng(seed + frame_index)
    output_frame = base_frame.copy()

    # Determine how many dinos to show, ensuring we don't try to show more than we have.
    num_dinos_to_show = min(2, len(dino_images))
    if num_dinos_to_show == 0:
        return base_frame

    # Pick unique dinosaurs to show for this frame.
    dino_indices = rng_dino.choice(len(dino_images), size=num_dinos_to_show, replace=False)

    for dino_index in dino_indices:
        dino_img_to_show = dino_images[dino_index]
        
        # Resize dino to a consistent, reasonable size
        scale_factor = rng_dino.uniform(0.3, 0.5)
        new_w = int(dino_img_to_show.shape[1] * scale_factor)
        new_h = int(dino_img_to_show.shape[0] * scale_factor)
        dino_resized = cv2.resize(dino_img_to_show, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Random position
        x_pos = rng_dino.integers(0, width - new_w)
        y_pos = rng_dino.integers(0, height - new_h)

        roi = output_frame[y_pos:y_pos+new_h, x_pos:x_pos+new_w]

        # --- Smart Blending based on image type ---
        if dino_resized.shape[2] == 4:
            # This is a transparent PNG (best quality)
            dino_bgr = dino_resized[:, :, :3]
            alpha = (dino_resized[:, :, 3] / 255.0) * DINO_OPACITY
            alpha_mask = alpha[:, :, np.newaxis]
            inv_alpha_mask = 1.0 - alpha_mask
            blended_roi = (dino_bgr * alpha_mask) + (roi * inv_alpha_mask)
        else:
            # This is a JPG or a non-transparent PNG.
            # It will appear as a semi-transparent rectangle.
            dino_bgr = dino_resized
            blended_roi = cv2.addWeighted(roi, 1.0 - DINO_OPACITY, dino_bgr, DINO_OPACITY, 0)
        
        output_frame[y_pos:y_pos+new_h, x_pos:x_pos+new_w] = blended_roi.astype(np.uint8)
    
    return output_frame


def get_glitched_frame(frame_index):
    """Generates the primary frame with all glitch/color effects."""
    event = next((e for e in timeline if e['start'] <= frame_index < e['end']), None)
    if not event: return base_image_bgr

    color_shifted_base = get_color_shifted_image(frame_index)
    
    if event['type'] == 'section':
        idx, p = event['index'], (frame_index-event['start'])/(event['end']-event['start'])
        if idx==0: frame=apply_base_effects(color_shifted_base, p, frame_index)
        elif idx==1: frame=apply_base_effects(color_shifted_base,1.0,frame_index)
        else:
            base = apply_base_effects(color_shifted_base,1.0,frame_index)
            if idx==2: frame=apply_heavy_glitch(base, p, frame_index)
            else:
                heavy = apply_heavy_glitch(base, 1.0, frame_index)
                bw=cv2.cvtColor(heavy, cv2.COLOR_BGR2GRAY)
                frame=cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
                if idx==4: # Destruction
                    pix_w,pix_h=max(1,int(lerp(width,16,p))),max(1,int(lerp(height,12,p)))
                    temp=cv2.resize(frame,(pix_w,pix_h),interpolation=cv2.INTER_NEAREST)
                    frame=(cv2.resize(temp,(width,height),interpolation=cv2.INTER_NEAREST)*(1.0-p)).astype(np.uint8)
                # Section 3 (B&W) just uses the frame as is
        if idx == 5: # Restoration - use original colors
            fade_start_frame = event['start'] + secs(0.5)
            if frame_index < fade_start_frame: return np.zeros_like(base_image_bgr)
            restore_p = (frame_index - fade_start_frame) / (event['end'] - fade_start_frame)
            frame = (base_image_bgr * restore_p).astype(np.uint8)
    
    elif event['type'] == 'burst':
        prev_sec_idx = event['index']
        # Determine base frame for explosion
        if prev_sec_idx < 4:
            base = apply_base_effects(color_shifted_base,1.0,frame_index)
            if prev_sec_idx > 0: base = apply_heavy_glitch(base,1.0,frame_index)
            if prev_sec_idx > 1: base = cv2.cvtColor(cv2.cvtColor(base,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        else: base = np.zeros_like(base_image_bgr) # After destruction/restore, explode from black
        burst_p = np.sin(((frame_index-event['start'])/(event['end']-event['start']))*np.pi)
        frame = apply_explosion(base, burst_p, event['seed'])

    return frame

def process_frame(frame_index):
    """
    This function is called by each worker. It generates the frame and writes it
    directly to the shared memory block, instead of returning it.
    """
    # Step 1: Generate the base frame with all glitch and color effects.
    glitched_frame = get_glitched_frame(frame_index)

    # Step 2: Check if a dinosaur overlay should be active.
    dino_event = next((e for e in dino_timeline if e['start'] <= frame_index < e['end']), None)
    
    final_frame = glitched_frame
    if dino_event:
        # Apply the overlay effect with constant opacity
        final_frame = apply_dino_overlay(glitched_frame, dino_event['seed'], frame_index)

    # Step 3: Convert the final frame to RGB for MoviePy.
    final_rgb_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
    
    # Step 4: Access shared memory and write the result.
    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
    results_array = np.ndarray(result_shape, dtype=result_dtype, buffer=existing_shm.buf)
    results_array[frame_index] = final_rgb_frame
    existing_shm.close()


if __name__ == "__main__":
    print("Starting combined effect video generation...")

    # --- Load Dinosaur Images ---
    loaded_dinos = []
    if not os.path.exists(DINO_DIR):
        print(f"INFO: Creating '{DINO_DIR}' directory. Please add transparent PNG dinosaur images here.")
        os.makedirs(DINO_DIR)
    else:
        print(f"INFO: Searching for images in '{DINO_DIR}' directory...")
        for fname in os.listdir(DINO_DIR):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    path = os.path.join(DINO_DIR, fname)
                    # Load the image, keeping its original channels (3 for JPG, 4 for transparent PNG)
                    dino_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if dino_img is not None:
                        loaded_dinos.append(dino_img)
                        if len(dino_img.shape) < 3 or dino_img.shape[2] == 3:
                            print(f"  > Loaded '{fname}' (JPG - will appear as a rectangle).")
                        elif dino_img.shape[2] == 4:
                            print(f"  > Loaded '{fname}' (Transparent PNG - best quality).")
                except Exception as e:
                    print(f"Warning: Could not load dino image '{fname}'. Reason: {e}")
    
    if not loaded_dinos:
        print("WARNING: No dinosaur images found or loaded. The dino overlay effect will be skipped.")

    # --- Generate Dino Timeline ---
    dino_timeline = []
    if loaded_dinos:
        current_dino_frame = 0
        while current_dino_frame < FRAME_COUNT:
            # Use fixed interval
            current_dino_frame += secs(DINO_INTERVAL_S)
            
            start = current_dino_frame
            end = start + secs(DINO_ONSCREEN_DURATION_S)
            
            if start >= FRAME_COUNT: break
            
            dino_timeline.append({
                'start': start,
                'end': end,
                'seed': rng.integers(0, 10000)
            })
            current_dino_frame = end


    try:
        source_image = cv2.imread(IMAGE_PATH)
        if source_image is None: raise FileNotFoundError
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Input image not found at '{IMAGE_PATH}'")
        print("-------------------------------------------------")
        print("Please check the 'IMAGE_PATH' variable in this script.")
        print("\nFiles found in the current directory:")
        try:
            files_in_dir = [f for f in os.listdir('.') if os.path.isfile(f)]
            if files_in_dir:
                for f in sorted(files_in_dir):
                    print(f"  - {f}")
            else:
                print("  (No files found)")
        except Exception as e:
            print(f"  (Could not list directory contents: {e})")
        print("-------------------------------------------------")
        exit()

    img_shape = source_image.shape
    h, w, _ = img_shape
    img_data = source_image.tobytes()
    dino_data_serial = [(d.tobytes(), d.shape) for d in loaded_dinos]

    # --- Setup Shared Memory ---
    result_shape = (FRAME_COUNT, h, w, 3)
    result_dtype = np.uint8
    shm_size = int(np.prod(result_shape) * np.dtype(result_dtype).itemsize)
    
    shm = None
    try:
        shm = shared_memory.SharedMemory(create=True, size=shm_size)
        print(f"Allocated {shm.size / 1e9:.2f} GB of shared memory.")
        
        num_processes = max(1, multiprocessing.cpu_count() - 4)
        print(f"Using {num_processes} processes to generate {FRAME_COUNT} frames...")

        init_args = (img_data, img_shape, dino_data_serial, dino_timeline, shm.name, result_shape, result_dtype)
        
        with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=init_args) as pool:
            # imap is now only used for its side-effects (writing to shared memory)
            # and to drive the tqdm progress bar.
            list(tqdm(pool.imap(process_frame, range(FRAME_COUNT)), total=FRAME_COUNT))

        # --- Assemble Video from Shared Memory ---
        # The main process can now access the completed frames directly.
        final_frames = np.ndarray(result_shape, dtype=result_dtype, buffer=shm.buf)
        
        output_filename = f"video_combo_effect_{int(time.time())}.mp4"
        print(f"\nAll frames generated. Assembling video: {output_filename}")

        clip = mp.ImageSequenceClip(list(final_frames), fps=FPS)
        clip.write_videofile(
            output_filename, codec='libx264', audio=False,
            preset='slow', ffmpeg_params=['-pix_fmt', 'yuv420p']
        )
        print("\nVideo generated successfully!")

    finally:
        # Ensure shared memory is always cleaned up
        if shm:
            shm.close()
            shm.unlink()
            print("Shared memory cleaned up.")