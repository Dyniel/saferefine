#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import cv2

def dice_bin(p, g):
    p = p.astype(np.uint8)
    g = g.astype(np.uint8)
    inter = int((p & g).sum())
    den = int(p.sum() + g.sum())
    if den == 0:
        return 1.0
    return float((2 * inter) / (den + 1e-6))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz_dir", required=True, help="folder z png 001__CpD*.png")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    p = Path(args.viz_dir)
    files = sorted(p.glob("*__CpD*.png"))
    if not files:
        raise SystemExit(f"Brak plików w {p}")

    # To działa na Twoich panelach:
    # - GT mask jest zwykle w górnym środku (czarne tło + żółty ring + czerwony cup)
    # - Pred mask jest zwykle w prawym górnym (czarne tło + coś)
    # Ponieważ to jest “render”, a nie surowa maska, zrobimy heurystykę:
    # wykryjemy cup jako czerwone piksele, disc jako żółte (ring+cup) w obu polach.
    def extract_masks(tile_bgr):
        hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)

        # RED (cup) – dwa zakresy
        red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
        cup = ((red1 > 0) | (red2 > 0))

        # YELLOW (disc ring) – zakres żółty
        yel = cv2.inRange(hsv, (18, 60, 80), (40, 255, 255))
        disc = (yel > 0) | cup  # disc = ring + cup (tak jak D)

        return disc, cup

    print(f"[cfg] viz_dir={p} files={len(files)}")
    show = min(args.n, len(files))

    for f in files[:show]:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]

        # układ 2x3 (z Twoich screenów): szerokość ~3 kafle, wysokość ~2 kafle
        # bierzemy:
        #  - GT tile: top-middle
        #  - Pred tile: top-right
        th, tw = H // 2, W // 3
        gt_tile = img[0:th, tw:2*tw]
        pr_tile = img[0:th, 2*tw:3*tw]

        gtD, gtC = extract_masks(gt_tile)
        prD, prC = extract_masks(pr_tile)

        dC = dice_bin(prC, gtC)
        dD = dice_bin(prD, gtD)
        cpd_prod = dC * dD
        cpd_sum = dC + dD

        print(f"{f.name} | dice_C={dC:.3f} dice_D={dD:.3f} | C*D={cpd_prod:.3f} C+D={cpd_sum:.3f}")

if __name__ == "__main__":
    main()
