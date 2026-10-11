"""Accuracy of multi/single auto-crop on synthetic flatbed scans (generic and anime themes).

    python benchmarks/synthetic_autocrop.py [scenario ...]

Prints mean/min IoU against ground truth, angle error, missed and extra
detections per scenario.  Set BENCH_OUT=<folder> to save overlays of the
worst cases.
"""
import os, sys, time, warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hasami.demo import synth_scan, SynthOptions
from hasami.core.autocrop import detect, AutoCropSettings, debug_overlay
from hasami.core.geometry import match_boxes, angle_difference
S = os.environ.get("BENCH_OUT", "")  # set to a folder to save overlays of failures
configs = {
 "basic": SynthOptions(bordered=0.0, light_photos=0.0),
 "bordered": SynthOptions(bordered=1.0, light_photos=0.0),
 "light": SynthOptions(bordered=0.3, light_photos=0.6),
 "black": SynthOptions(background="black"),
 "close": SynthOptions(n_photos=5, min_gap=4, photo_size=(0.25, 0.35)),
 "touch": SynthOptions(n_photos=4, min_gap=0, photo_size=(0.25, 0.35), max_angle=10),
 "corner+strip": SynthOptions(corner_photo=True, frame_strip=True),
 "jpeg": SynthOptions(jpeg_quality=70, noise=3.0),
 "document": SynthOptions(document=True),
 "anime": SynthOptions(theme="anime"),
 "anime-bordered": SynthOptions(theme="anime", bordered=0.8),
 "anime-light": SynthOptions(theme="anime", light_photos=0.6),
 "anime-black": SynthOptions(theme="anime", background="black"),
 "anime-close": SynthOptions(theme="anime", n_photos=5, min_gap=4, photo_size=(0.25, 0.35)),
 "anime-print": SynthOptions(theme="anime", document=True),
}
which = sys.argv[1:] or list(configs)
N = 12
for name in which:
    o = configs[name]
    ious, angs, miss, extra, t = [], [], 0, 0, 0
    for seed in range(N):
        sc = synth_scan(o, seed=seed)
        st = AutoCropSettings(mode="single" if o.document else "multi", margin_mm=0.0)
        t0 = time.time(); boxes = detect(sc.image, st, dpi=200); t += time.time()-t0
        m = match_boxes(boxes, sc.truth)
        good = [(i,j,iou) for i,j,iou in m if iou > 0.5]
        miss += len(sc.truth) - len(good); extra += len(boxes) - len(good)
        for i,j,iou in good:
            ious.append(iou); angs.append(angle_difference(boxes[i].angle, sc.truth[j].angle))
        worst = min([iou for _,_,iou in good], default=0)
        if S and (len(good) != len(sc.truth) or len(boxes) != len(good) or worst < 0.97):
            ov = debug_overlay(sc.image, boxes, sc.truth)
            cv2.imwrite(os.path.join(S, f"fail_{name}_{seed}.jpg"), cv2.cvtColor(cv2.resize(ov,None,fx=0.4,fy=0.4), cv2.COLOR_RGB2BGR))
    ious = np.array(ious); angs=np.array(angs)
    print(f"{name:14s} mean IoU {ious.mean():.4f} min {ious.min() if len(ious) else 0:.4f}  ang mean {angs.mean():.3f} max {angs.max() if len(angs) else 0:.3f}  miss {miss} extra {extra}  {t/N*1000:.0f} ms/img")
