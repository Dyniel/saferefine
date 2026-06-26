#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(root):
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])


def stem_key(path):
    s = path.stem
    for suffix in ("_mask", "-mask", "_segmentation", "_lesion", "_seg", "_label", "_gt"):
        if s.lower().endswith(suffix):
            return s[: -len(suffix)]
    return s


def read_rgb(path, size):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size > 0:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def read_mask(path, size, threshold, invert):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    if size > 0:
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    y = (m > threshold).astype(np.uint8)
    if invert:
        y = 1 - y
    return y


def split_ids(n, train_frac, val_frac, seed):
    ids = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(max(1, n_train), max(1, n - 2))
    n_val = min(max(1, n_val), max(1, n - n_train - 1))
    train = sorted(ids[:n_train])
    val = sorted(ids[n_train : n_train + n_val])
    test = sorted(ids[n_train + n_val :])
    if not test:
        test = val[-1:]
        val = val[:-1]
    return train, val, test


def write_split(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(i) for i in ids) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--name", default="binary_dataset")
    ap.add_argument("--img_size", type=int, default=352)
    ap.add_argument("--threshold", type=int, default=127)
    ap.add_argument("--invert", type=int, default=0)
    ap.add_argument("--compress", type=int, default=1)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    images = list_images(args.image_dir)
    masks = list_images(args.mask_dir)
    mask_by_key = {stem_key(p): p for p in masks}
    pairs = []
    missing = []
    for img in images:
        key = stem_key(img)
        mask = mask_by_key.get(key)
        if mask is None:
            missing.append(str(img))
            continue
        pairs.append((key, img, mask))

    if not pairs:
        raise RuntimeError(f"No image/mask pairs found: image_dir={args.image_dir} mask_dir={args.mask_dir}")

    xs, ys, ids = [], [], []
    for key, img_p, mask_p in pairs:
        xs.append(read_rgb(img_p, args.img_size))
        ys.append(read_mask(mask_p, args.img_size, args.threshold, bool(args.invert)))
        ids.append(key)

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_npz = np.savez_compressed if args.compress else np.savez
    save_npz(out_npz, images=np.stack(xs, axis=0), masks=np.stack(ys, axis=0), ids=np.asarray(ids, dtype=object))

    train, val, test = split_ids(len(ids), args.train_frac, args.val_frac, args.seed)
    split_dir = Path(args.split_dir)
    write_split(split_dir / "train.txt", train)
    write_split(split_dir / "val.txt", val)
    write_split(split_dir / "test.txt", test)

    manifest = {
        "name": args.name,
        "out_npz": str(out_npz),
        "split_dir": str(split_dir),
        "n": len(ids),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "img_size": args.img_size,
        "compressed": bool(args.compress),
        "missing_images_without_masks": len(missing),
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
