#!/usr/bin/env python3

"""
Lunar Rover Autonomy Demo - Visual Dashboard

Same pipeline as lunar_demo.py, but renders a 2x2 panel instead of only
printing to the terminal:

    +----------------------+----------------------+
    |   Rover Camera Image | Terrain Segmentation  |
    +----------------------+----------------------+
    |   Cost Map            |  LingBot Memory Map  |
    +----------------------+----------------------+
    Mission decision text printed below the panel.
    
Usage:
    python3 lunar_demo_dashboard.py                  # save PNGs, no live window
    python3 lunar_demo_dashboard.py --show            # also open a live cv2 window
                                                       # (press any key to advance,
                                                       # 'q' to quit)
    python3 lunar_demo_dashboard.py --limit 5          # only process first 5 images
"""

import argparse
import os
import sys

import cv2
import numpy as np

from rl_deploy import TerrainSegmenter, DummySegmenter
from terrain_costmap import TerrainCostMapper
from science_targeting import RockTargetSelector


# ----------------------------
# Configuration
# ----------------------------

IMAGE_DIR = "/home/jetson/rl_deploy/demo_images"
SEG_MODEL = "/home/jetson/rl_deploy/mark_model/unet_lunar_segmentation.pth"
OUTPUT_DIR = "/home/jetson/rl_deploy/dashboard_output"

PANEL_CELL_W = 480
PANEL_CELL_H = 480
STRIP_H = 90  # bottom text strip for the mission decision

# Corrected 5-class mapping (0=ground,1=big rock,2=small rock,3=crater,4=sky)
# - matches TerrainSegmenter.CLASS_* after the mapping fix.
CLASS_COLORS_BGR = {
    0: (110, 170, 200),   # ground/regolith - tan
    1: (0, 0, 200),       # big rock - red
    2: (0, 140, 255),     # small rock - orange
    3: (200, 0, 200),     # crater - magenta
    4: (235, 206, 135),   # sky - light blue
}
CLASS_NAMES = {0: "ground", 1: "big rock", 2: "small rock", 3: "crater", 4: "sky"}


def colorize_mask(mask):
    """Map a (H, W) class-index mask to a BGR image using CLASS_COLORS_BGR."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS_BGR.items():
        out[mask == cls_id] = color
    return out


def make_legend(width, height):
    legend = np.full((height, width, 3), 30, dtype=np.uint8)
    y = 20
    for cls_id in sorted(CLASS_COLORS_BGR):
        color = CLASS_COLORS_BGR[cls_id]
        cv2.rectangle(legend, (10, y - 12), (28, y + 2), color, -1)
        cv2.putText(legend, CLASS_NAMES[cls_id], (36, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return legend


def fit_cell(img, w=PANEL_CELL_W, h=PANEL_CELL_H, label=""):
    """Resize an image (or make a blank placeholder if None) to one panel
    cell and stamp a title in the corner."""
    if img is None:
        cell = np.full((h, w, 3), 40, dtype=np.uint8)
    else:
        cell = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(cell, (0, 0), (w - 1, 24), (20, 20, 20), -1)
    cv2.putText(cell, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def placeholder_panel(w, h, lines):
    panel = np.full((h, w, 3), 25, dtype=np.uint8)
    y = h // 2 - (len(lines) * 18) // 2
    for line in lines:
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x = max(8, (w - size[0]) // 2)
        cv2.putText(panel, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180), 1, cv2.LINE_AA)
        y += 22
    return panel


def build_panel(rgb, seg_color, cost_color, lingbot_panel, decision_text, obs_label):
    top = np.hstack([
        fit_cell(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), label=f"{obs_label} - Rover Camera"),
        fit_cell(seg_color, label="Terrain Segmentation"),
    ])
    bottom = np.hstack([
        fit_cell(cost_color, label="Cost Map (perspective-space, no depth)"),
        fit_cell(lingbot_panel, label="LingBot Memory Map"),
    ])
    grid = np.vstack([top, bottom])

    strip = np.full((STRIP_H, grid.shape[1], 3), 15, dtype=np.uint8)
    for i, line in enumerate(decision_text.split("\n")):
        cv2.putText(strip, line, (12, 26 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([grid, strip])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true",
                         help="Open a live cv2 window (press any key to advance, q to quit)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N images")
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--model-path", default=SEG_MODEL)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--lingbot-model", default=None,
                         help="Path to LingBot-Map checkpoint. If set (with "
                              "--lingbot-dir), the dashboard actually loads "
                              "LingBot-Map and shows its real init/batch "
                              "status in the memory panel.")
    parser.add_argument("--lingbot-dir", default=None,
                         help="Path where the lingbot-map repo was cloned.")
    parser.add_argument("--lingbot-workdir", default=None,
                         help="Scratch dir for LingBot keyframes/batches "
                              "(default: ./lingbot_scratch next to this script)")
    args = parser.parse_args()

    print("\n=== LUNAR ROVER DASHBOARD DEMO STARTING ===\n")

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        segmenter = TerrainSegmenter(args.model_path)
    except Exception as e:
        print(f"[WARN] Could not load real segmenter ({e}); using DummySegmenter. "
              f"Segmentation/cost panels will be placeholders.")
        segmenter = DummySegmenter()

    science = RockTargetSelector(preferred_class="big")
    cost_mapper = TerrainCostMapper(map_resolution_m_per_px=0.05, rover_radius_m=0.25)

    # --- Real LingBot-Map init (only if you passed the flags) ---
    # This is genuine LingBot code executing on this hardware, not a mock.
    # It does NOT force full reconstruction to succeed — the panel just
    # honestly reports whatever state it's actually in.
    memory_mapper = None
    lingbot_init_error = None
    if args.lingbot_model and args.lingbot_dir:
        try:
            from lingbot_memory_map import LingbotMemoryMapper
            workdir = args.lingbot_workdir or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "lingbot_scratch")
            memory_mapper = LingbotMemoryMapper(
                model_path=args.lingbot_model,
                lingbot_dir=args.lingbot_dir,
                workdir=workdir,
            )
            print("[OK] LingBot-Map memory mapper initialized")
        except Exception as e:
            lingbot_init_error = str(e)
            print(f"[WARN] LingBot-Map memory mapper unavailable: {e}")

    legend = make_legend(PANEL_CELL_W, PANEL_CELL_H)

    images = sorted([
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ])
    if args.limit:
        images = images[:args.limit]

    print(f"Loaded {len(images)} lunar observations\n")

    for i, path in enumerate(images):
        obs_label = f"OBSERVATION {i + 1}"
        print("=" * 50)
        print(obs_label)
        print(path)

        image = cv2.imread(path)
        if image is None:
            print("Could not load image")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ---- Terrain understanding ----
        overall, left, right = segmenter.analyze(rgb)
        mask = getattr(segmenter, "_last_mask", None)

        print("\nTERRAIN ANALYSIS")
        print(f"Overall feasibility: {overall:.2f}")
        print(f"Left terrain:       {left:.2f}")
        print(f"Right terrain:      {right:.2f}")

        if mask is not None:
            seg_color = colorize_mask(mask)
            cost_grid = cost_mapper.generate(mask)
            cost_color = cv2.applyColorMap(cost_grid, cv2.COLORMAP_JET)
        else:
            seg_color = placeholder_panel(PANEL_CELL_W, PANEL_CELL_H,
                                           ["segmentation unavailable"])
            cost_color = placeholder_panel(PANEL_CELL_W, PANEL_CELL_H,
                                            ["cost map unavailable"])

        # ---- Science targeting ----
        target = science.find_target(mask, None) if mask is not None else None

        print("\nSCIENCE")
        if target:
            print("Target found!")
            print(f"Type: {target.class_name}")
            print(f"Location: {target.center_px}")
            print(f"Score: {target.score:.2f}")
        else:
            print("No science target found")

        # ---- Mission decision ----
        print("\nMISSION DECISION")
        if target:
            decision = "Approach science target"
        elif overall < 0.4:
            decision = "Avoid hazardous terrain"
        else:
            decision = "Continue exploration"
        print(f"-> {decision}")

        decision_lines = [f"Feasibility: {overall:.2f} (L {left:.2f} / R {right:.2f})"]
        if target:
            decision_lines.append(
                f"Science target: {target.class_name} @ {target.center_px} "
                f"(score {target.score:.2f})"
            )
        decision_lines.append(f"Mission decision: -> {decision}")
        decision_text = "\n".join(decision_lines)

        # ---- LingBot memory panel: real status, not a mock ----
        if memory_mapper is not None:
            # SYNTHETIC odometry: these demo images have no real trajectory,
            # so this is just a steadily-incrementing x to exercise the
            # keyframe/batch pipeline. Not a real drive, and the panel says so.
            memory_mapper.maybe_capture(i, rgb, i * 0.5, 0.0, 0.0)
            status_lines = [
                "LingBot-Map: mapper object built",
                "(model_path/repo not verified until",
                "the first batch attempt runs - see",
                "Batch status below for real proof)",
                f"Batch status: {memory_mapper.last_batch_status}",
            ]
            if memory_mapper.last_batch_error:
                status_lines.append(f"Last error: {memory_mapper.last_batch_error[:60]}")
            status_lines.append("(odometry fed to it is SYNTHETIC -")
            status_lines.append("these images aren't a real trajectory)")
            lingbot_panel = placeholder_panel(PANEL_CELL_W, PANEL_CELL_H, status_lines)
        elif lingbot_init_error:
            lingbot_panel = placeholder_panel(
                PANEL_CELL_W, PANEL_CELL_H,
                ["LingBot-Map: init failed", lingbot_init_error[:70]],
            )
        else:
            lingbot_panel = placeholder_panel(
                PANEL_CELL_W, PANEL_CELL_H,
                [
                    "LingBot-Map not configured for",
                    "this run. Pass --lingbot-model",
                    "and --lingbot-dir to load it for real.",
                ],
            )

        panel = build_panel(rgb, seg_color, cost_color, lingbot_panel,
                             decision_text, obs_label)

        out_path = os.path.join(args.output_dir, f"panel_{i + 1:03d}.png")
        cv2.imwrite(out_path, panel)

        if args.show:
            cv2.imshow("Lunar Rover Dashboard", panel)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()

    print(f"\n=== DEMO COMPLETE - panels saved to {args.output_dir} ===")


if __name__ == "__main__":
    main()
