#!/usr/bin/env python3
"""
Author: Abhinav Rajaram
"""
import cv2
import time
import cProfile
import pstats
from utils.greenonbrown import GreenOnBrown
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 1 | CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# 0 = default camera, or "field_clip.mp4"
VIDEO_SRC = "datasets/videos/green_on_fallow.mp4"
WARMUP_FRM = 30                   # ignore first N frames for stabilisation
MAX_FRAMES = 300                  # timed frames; set None for “run until EOF”
SAVE_OUT = True               # True = write annotated video
OUT_FILE = "datasets/videos_output/bench_output.mp4"   # if SAVE_OUT is True

PARAMS = {                         # your ExG thresholds & options
    'exg_min': 30, 'exg_max': 250,
    'hue_min': 30, 'hue_max': 90,
    'brightness_min': 5,  'brightness_max': 200,
    'saturation_min': 30, 'saturation_max': 255,
    'min_detection_area': 100,
    'show_display': True,         # GUI adds overhead; keep False for timing
    'algorithm': 'exg',
    'invert_hue': False,
    'label': 'WEED'
}

# ──────────────────────────────────────────────────────────────────────────────
# 2 | BENCHMARK FUNCTION
# ──────────────────────────────────────────────────────────────────────────────


def run_benchmark():
    algo = GreenOnBrown(algorithm='exg')

    cap = cv2.VideoCapture(VIDEO_SRC, cv2.CAP_ANY)
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {VIDEO_SRC}")

    # prepare video writer if required
    writer = None
    if SAVE_OUT:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(OUT_FILE, fourcc, fps_in, (width, height))

    # — Warm‑up — -------------------------------------------------------------
    for _ in range(WARMUP_FRM):
        ok, frame = cap.read()
        if not ok:
            break
        algo.inference(frame, **PARAMS)

    # — Timed loop — ----------------------------------------------------------
    total_time = 0.0
    counted = 0
    tick_start = time.perf_counter()

    while True:
        if MAX_FRAMES and counted >= MAX_FRAMES:
            break

        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        contours, boxes, centres, annotated = algo.inference(frame, **PARAMS)
        t1 = time.perf_counter()
        total_time += (t1 - t0)
        counted += 1

        if SAVE_OUT and writer is not None:
            writer.write(annotated)

        # optional live FPS overlay (not used for timing)
        if PARAMS['show_display']:
            fps_now = 1.0 / (t1 - t0)
            cv2.putText(annotated, f"{fps_now:5.1f} FPS",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("GreenOnBrown FPS test", annotated)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

    tick_end = time.perf_counter()
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # — Results — -------------------------------------------------------------
    mean_fps = counted / total_time if total_time else 0
    wall_clock = tick_end - tick_start
    print("───────────────────────── BENCHMARK SUMMARY ─────────────────────────")
    print(f"Frames processed   : {counted}")
    print(f"Algorithm time     : {total_time:8.3f} s")
    print(f"Mean per‑frame     : {1000*total_time/counted:8.2f} ms")
    print(f"Mean FPS (algo)    : {mean_fps:8.2f}")
    print(f"Total wall‑clock   : {wall_clock:8.3f} s (incl. capture/GUI)")
    print("───────────────────────────────────────────────────────────────────────")


# ──────────────────────────────────────────────────────────────────────────────
# 3 | PROFILING ENTRY‑POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with cProfile.Profile() as pr:
        run_benchmark()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(15)                # Top‑15 most expensive calls
    stats.dump_stats("profile_results.prof")
