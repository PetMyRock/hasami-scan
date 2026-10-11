"""Accuracy of booklet mode on synthetic open booklets too wide for the bed.

    python benchmarks/synthetic_booklet.py [scenario ...]

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
from hasami.demo import synth_booklet, BookletOptions
from hasami.core.autocrop import detect, AutoCropSettings, debug_overlay
from hasami.core.geometry import match_boxes, angle_difference
S = os.environ.get("BENCH_OUT", "")  # set to a folder to save overlays of failures
configs = {
 "white": BookletOptions(),
 "dark-lid": BookletOptions(background="dark"),
 "horizontal": BookletOptions(spine="horizontal"),
 "flat": BookletOptions(gutter_depth=(0.0, 0.03), fold_line=0.0, staples=0.0),
 "deep-gutter": BookletOptions(gutter_depth=(0.4, 0.6), gutter_width=(0.03, 0.08)),
 "jpeg": BookletOptions(jpeg_quality=70, noise=3.0),
 "full-spread": BookletOptions(fully_visible=True),
 "artbook": BookletOptions(content="artbook"),
 "artbook-dark": BookletOptions(content="artbook", background="dark"),
 "artbook-flat": BookletOptions(content="artbook", gutter_depth=(0.0, 0.04), fold_line=0.0, staples=0.0),
 "magazine": BookletOptions(content="mixed"),
 "magazine-horiz": BookletOptions(content="mixed", spine="horizontal"),
}
which = sys.argv[1:] or list(configs)
N = 12
for name in which:
    o = configs[name]
    ious, angs, wrong, t = [], [], 0, 0
    for seed in range(N):
        sc = synth_booklet(o, seed=seed)
        t0 = time.time(); boxes = detect(sc.image, AutoCropSettings(mode="booklet", margin_mm=0), dpi=200); t += time.time()-t0
        m = match_boxes(boxes, sc.truth)
        good = [(i,j,iou) for i,j,iou in m if iou > 0.5]
        ok_count = len(boxes) == len(sc.truth) and len(good) == len(sc.truth)
        if not ok_count: wrong += 1
        for i,j,iou in good:
            ious.append(iou); angs.append(angle_difference(boxes[i].angle, sc.truth[j].angle))
        worst = min([iou for _,_,iou in good], default=0)
        if S and (not ok_count or worst < 0.98):
            ov = debug_overlay(sc.image, boxes, sc.truth)
            cv2.imwrite(os.path.join(S, f"bfail_{name}_{seed}.jpg"), cv2.cvtColor(cv2.resize(ov,None,fx=0.35,fy=0.35), cv2.COLOR_RGB2BGR))
    ious = np.array(ious) if ious else np.zeros(1)
    angs = np.array(angs) if angs else np.zeros(1)
    print(f"{name:12s} mean IoU {ious.mean():.4f} min {ious.min():.4f}  ang max {angs.max():.3f}  wrong-count {wrong}/{N}  {t/N*1000:.0f} ms/img")
