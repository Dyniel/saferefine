#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np


def write_split(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(i)) for i in ids) + "\n")


def normalize_to_uint8(volume):
    finite = np.asarray(volume[np.isfinite(volume)], dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.uint8)
    x = np.clip((volume - lo) / (hi - lo), 0.0, 1.0)
    return (255.0 * x + 0.5).astype(np.uint8)


def resize_slice(img, mask, size):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, (mask > 0).astype(np.uint8)


def patient_split(patients, train_frac, val_frac, seed):
    rng = random.Random(seed)
    pts = list(patients)
    rng.shuffle(pts)
    n = len(pts)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(max(1, n_train), max(1, n - 2))
    n_val = min(max(1, n_val), max(1, n - n_train - 1))
    train = sorted(pts[:n_train])
    val = sorted(pts[n_train:n_train + n_val])
    test = sorted(pts[n_train + n_val:])
    return train, val, test


def collect_slices(root, img_size, min_mask_pixels, margin_slices):
    root = Path(root)
    image_paths = sorted((root / "imagesTr").glob("la_*.nii.gz"))
    rows = []
    for img_path in image_paths:
        label_path = root / "labelsTr" / img_path.name
        if not label_path.exists():
            continue
        patient = img_path.name.replace(".nii.gz", "")
        img = nib.load(str(img_path)).get_fdata(dtype=np.float32)
        lab = nib.load(str(label_path)).get_fdata(dtype=np.float32)
        if img.shape != lab.shape:
            raise RuntimeError(f"shape mismatch: {img_path} {img.shape} vs {label_path} {lab.shape}")
        img8 = normalize_to_uint8(img)
        mask = lab > 0
        z_has = np.flatnonzero(mask.reshape(-1, mask.shape[-1]).sum(axis=0) >= int(min_mask_pixels))
        if len(z_has) == 0:
            continue
        z0 = max(0, int(z_has.min()) - int(margin_slices))
        z1 = min(mask.shape[-1] - 1, int(z_has.max()) + int(margin_slices))
        for z in range(z0, z1 + 1):
            im2 = img8[:, :, z]
            ma2 = mask[:, :, z].astype(np.uint8)
            if ma2.sum() < int(min_mask_pixels) and z not in z_has:
                # Keep only immediate margin slices; they are useful context but
                # should not dominate the dataset.
                pass
            im2, ma2 = resize_slice(im2, ma2, img_size)
            rgb = np.repeat(im2[..., None], 3, axis=-1)
            rows.append({
                "patient": patient,
                "z": int(z),
                "image": rgb,
                "mask": ma2,
                "id": f"{patient}_z{z:03d}",
                "mask_pixels": int(ma2.sum()),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="raw_data/msd/Task02_Heart")
    ap.add_argument("--out_npz", default="data_npz/msd_heart_mri_352.npz")
    ap.add_argument("--split_dir", default="dataset_splits/msd_heart_mri_352")
    ap.add_argument("--img_size", type=int, default=352)
    ap.add_argument("--min_mask_pixels", type=int, default=16)
    ap.add_argument("--margin_slices", type=int, default=2)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rows = collect_slices(args.root, args.img_size, args.min_mask_pixels, args.margin_slices)
    if not rows:
        raise RuntimeError(f"No slices collected from {args.root}")
    patients = sorted({r["patient"] for r in rows})
    train_p, val_p, test_p = patient_split(patients, args.train_frac, args.val_frac, args.seed)
    split_by_patient = {"train": set(train_p), "val": set(val_p), "test": set(test_p)}
    split_indices = {k: [] for k in split_by_patient}
    for i, row in enumerate(rows):
        for split, pts in split_by_patient.items():
            if row["patient"] in pts:
                split_indices[split].append(i)
                break

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        images=np.stack([r["image"] for r in rows], axis=0).astype(np.uint8),
        masks=np.stack([r["mask"] for r in rows], axis=0).astype(np.uint8),
        ids=np.asarray([r["id"] for r in rows], dtype=object),
        patients=np.asarray([r["patient"] for r in rows], dtype=object),
        z=np.asarray([r["z"] for r in rows], dtype=np.int16),
    )

    split_dir = Path(args.split_dir)
    write_split(split_dir / "train.txt", split_indices["train"])
    write_split(split_dir / "val.txt", split_indices["val"])
    write_split(split_dir / "test.txt", split_indices["test"])
    manifest = {
        "name": "msd_heart_mri",
        "source": "Medical Segmentation Decathlon Task02 Heart",
        "modality": "MRI",
        "target": "left atrium",
        "license": "CC-BY-SA 4.0",
        "root": str(Path(args.root).resolve()),
        "out_npz": str(out_npz.resolve()),
        "split_dir": str(split_dir.resolve()),
        "img_size": int(args.img_size),
        "n_slices": len(rows),
        "n_patients": len(patients),
        "patients_train": train_p,
        "patients_val": val_p,
        "patients_test": test_p,
        "n_train": len(split_indices["train"]),
        "n_val": len(split_indices["val"]),
        "n_test": len(split_indices["test"]),
        "min_mask_pixels": int(args.min_mask_pixels),
        "margin_slices": int(args.margin_slices),
        "seed": int(args.seed),
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
