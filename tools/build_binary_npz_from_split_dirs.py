#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path

import numpy as np

from build_binary_npz_from_dirs import list_images, read_mask, read_rgb, stem_key, write_split


def paired_split(image_dir, mask_dir):
    images = list_images(image_dir)
    masks = list_images(mask_dir)
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
    return pairs, missing


def add_optional_split(args, split_name, all_pairs, missing_by_split):
    image_dir = getattr(args, f"{split_name}_image_dir")
    mask_dir = getattr(args, f"{split_name}_mask_dir")
    if not image_dir and not mask_dir:
        return []
    if not image_dir or not mask_dir:
        raise RuntimeError(f"Both --{split_name}_image_dir and --{split_name}_mask_dir are required.")
    pairs, missing = paired_split(image_dir, mask_dir)
    if not pairs:
        raise RuntimeError(f"No {split_name} image/mask pairs found: image_dir={image_dir} mask_dir={mask_dir}")
    missing_by_split[split_name] = len(missing)
    start = len(all_pairs)
    all_pairs.extend(pairs)
    return list(range(start, start + len(pairs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_image_dir", required=True)
    ap.add_argument("--train_mask_dir", required=True)
    ap.add_argument("--val_image_dir")
    ap.add_argument("--val_mask_dir")
    ap.add_argument("--test_image_dir")
    ap.add_argument("--test_mask_dir")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--name", default="binary_dataset")
    ap.add_argument("--img_size", type=int, default=352)
    ap.add_argument("--threshold", type=int, default=127)
    ap.add_argument("--invert", type=int, default=0)
    ap.add_argument("--val_from_train_frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--compress", type=int, default=1)
    args = ap.parse_args()

    all_pairs = []
    missing_by_split = {}
    train_idx = add_optional_split(args, "train", all_pairs, missing_by_split)
    val_idx = add_optional_split(args, "val", all_pairs, missing_by_split)
    test_idx = add_optional_split(args, "test", all_pairs, missing_by_split)
    if not val_idx and args.val_from_train_frac > 0:
        if not 0 < args.val_from_train_frac < 1:
            raise RuntimeError("--val_from_train_frac must be between 0 and 1.")
        shuffled = list(train_idx)
        random.Random(args.seed).shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * args.val_from_train_frac)))
        val_idx = sorted(shuffled[:n_val])
        train_idx = sorted(shuffled[n_val:])

    xs, ys, ids = [], [], []
    for split, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for i in indices:
            key, img_p, mask_p = all_pairs[i]
            xs.append(read_rgb(img_p, args.img_size))
            ys.append(read_mask(mask_p, args.img_size, args.threshold, bool(args.invert)))
            ids.append(f"{split}/{key}")

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_npz = np.savez_compressed if args.compress else np.savez
    save_npz(out_npz, images=np.stack(xs, axis=0), masks=np.stack(ys, axis=0), ids=np.asarray(ids, dtype=object))

    split_dir = Path(args.split_dir)
    write_split(split_dir / "train.txt", train_idx)
    if val_idx:
        write_split(split_dir / "val.txt", val_idx)
    if test_idx:
        write_split(split_dir / "test.txt", test_idx)

    manifest = {
        "name": args.name,
        "out_npz": str(out_npz),
        "split_dir": str(split_dir),
        "n": len(ids),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "img_size": args.img_size,
        "compressed": bool(args.compress),
        "missing_images_without_masks": missing_by_split,
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
