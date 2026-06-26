#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np

from build_binary_npz_from_dirs import read_mask, read_rgb, write_split
from build_binary_npz_from_split_dirs import add_optional_split


def collect_pairs(args):
    all_pairs = []
    missing_by_split = {}
    train_idx = add_optional_split(args, "train", all_pairs, missing_by_split)
    val_idx = add_optional_split(args, "val", all_pairs, missing_by_split)
    test_idx = add_optional_split(args, "test", all_pairs, missing_by_split)

    ids = []
    for split, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for i in indices:
            key, _, _ = all_pairs[i]
            ids.append(f"{split}/{key}")

    return all_pairs, ids, train_idx, val_idx, test_idx, missing_by_split


def build_chunk(args):
    all_pairs, ids, *_ = collect_pairs(args)
    n = len(all_pairs)
    start = max(0, int(args.start))
    end = min(n, int(args.end))
    if start >= end:
        raise RuntimeError(f"Empty chunk range: start={start} end={end} n={n}")

    xs, ys, chunk_ids = [], [], []
    for i in range(start, end):
        _, img_p, mask_p = all_pairs[i]
        xs.append(read_rgb(img_p, args.img_size))
        ys.append(read_mask(mask_p, args.img_size, args.threshold, bool(args.invert)))
        chunk_ids.append(ids[i])

    out_dir = Path(args.chunk_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"chunk_{start:06d}_{end:06d}.npz"
    np.savez(
        out_path,
        images=np.stack(xs, axis=0),
        masks=np.stack(ys, axis=0),
        ids=np.asarray(chunk_ids, dtype=object),
        start=np.asarray(start, dtype=np.int64),
        end=np.asarray(end, dtype=np.int64),
    )
    print(json.dumps({"chunk": str(out_path), "start": start, "end": end, "n": end - start}, sort_keys=True), flush=True)


def write_plan(args):
    all_pairs, ids, train_idx, val_idx, test_idx, missing_by_split = collect_pairs(args)
    n = len(all_pairs)
    chunk_size = int(args.chunk_size)
    if chunk_size <= 0:
        raise RuntimeError("--chunk_size must be positive")
    chunks = []
    for start in range(0, n, chunk_size):
        chunks.append({"start": start, "end": min(n, start + chunk_size)})
    plan = {
        "name": args.name,
        "n": n,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "img_size": args.img_size,
        "chunk_size": chunk_size,
        "chunks": chunks,
        "ids": ids,
        "missing_images_without_masks": missing_by_split,
    }
    chunk_dir = Path(args.chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    print(json.dumps({k: plan[k] for k in ("name", "n", "n_train", "n_val", "n_test", "img_size", "chunk_size")}, sort_keys=True), flush=True)


def merge_chunks(args):
    chunk_dir = Path(args.chunk_dir)
    plan = json.loads((chunk_dir / "plan.json").read_text())
    images, masks, ids = [], [], []
    for ch in plan["chunks"]:
        p = chunk_dir / f"chunk_{ch['start']:06d}_{ch['end']:06d}.npz"
        if not p.exists():
            raise RuntimeError(f"Missing chunk: {p}")
        z = np.load(p, allow_pickle=True)
        if int(z["start"]) != ch["start"] or int(z["end"]) != ch["end"]:
            raise RuntimeError(f"Chunk range mismatch: {p}")
        images.append(z["images"])
        masks.append(z["masks"])
        ids.append(z["ids"])

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_npz = np.savez_compressed if args.compress else np.savez
    save_npz(
        out_npz,
        images=np.concatenate(images, axis=0),
        masks=np.concatenate(masks, axis=0),
        ids=np.concatenate(ids, axis=0),
    )

    split_dir = Path(args.split_dir)
    n_train = int(plan["n_train"])
    n_val = int(plan["n_val"])
    n_test = int(plan["n_test"])
    write_split(split_dir / "train.txt", list(range(0, n_train)))
    write_split(split_dir / "val.txt", list(range(n_train, n_train + n_val)))
    write_split(split_dir / "test.txt", list(range(n_train + n_val, n_train + n_val + n_test)))
    manifest = {
        "name": args.name,
        "out_npz": str(out_npz),
        "split_dir": str(split_dir),
        "n": int(plan["n"]),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "img_size": int(plan["img_size"]),
        "compressed": bool(args.compress),
        "chunk_dir": str(chunk_dir),
        "missing_images_without_masks": plan["missing_images_without_masks"],
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("plan", "chunk", "merge"), required=True)
    ap.add_argument("--train_image_dir", required=True)
    ap.add_argument("--train_mask_dir", required=True)
    ap.add_argument("--val_image_dir")
    ap.add_argument("--val_mask_dir")
    ap.add_argument("--test_image_dir")
    ap.add_argument("--test_mask_dir")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--chunk_dir", required=True)
    ap.add_argument("--name", default="binary_dataset")
    ap.add_argument("--img_size", type=int, default=352)
    ap.add_argument("--threshold", type=int, default=127)
    ap.add_argument("--invert", type=int, default=0)
    ap.add_argument("--compress", type=int, default=0)
    ap.add_argument("--chunk_size", type=int, default=400)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "plan":
        write_plan(args)
    elif args.mode == "chunk":
        build_chunk(args)
    else:
        merge_chunks(args)


if __name__ == "__main__":
    main()
