#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import time
import math
import json
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _swap12_mask012(a: np.ndarray) -> np.ndarray:
    # a: uint8 with labels {0,1,2}
    out = a.copy()
    out[a == 1] = 255
    out[a == 2] = 1
    out[out == 255] = 2
    return out


def _clahe_rgb(img_rgb: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def _center_square(img: np.ndarray) -> tuple[np.ndarray, tuple[int,int,int,int]]:
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0:y0+s, x0:x0+s], (x0, y0, x0+s, y0+s)


def _resize(img: np.ndarray, size: int, interp) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=interp)


def _read_image_rgb(p: Path) -> np.ndarray:
    im = np.array(Image.open(p).convert("RGB"))
    return im


def _infer_file_list(in_path: Path, limit: int = 0) -> list[Path]:
    # Accept either dataset root with Images/, or a directory directly containing images
    cand_dirs = []
    if (in_path / "Images").is_dir():
        cand_dirs.append(in_path / "Images")
    cand_dirs.append(in_path)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = []
    for d in cand_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
        if files:
            break

    if limit and limit > 0:
        files = files[:limit]
    return files


def _import_model_module(model_py: str):
    mp = Path(model_py)
    if not mp.exists():
        raise FileNotFoundError(f"model_py not found: {model_py}")
    spec = importlib.util.spec_from_file_location(mp.stem, str(mp))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model module from: {model_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _load_model(mod, ckpt_path: str, device: str):
    """
    Supported patterns in model_py:
      A) class GraphSeg(torch.nn.Module): can be instantiated with no args;
         then either:
           - has .load_ckpt(path, device=...)
           - or accepts ckpt_path in __init__(ckpt_path=...)
      B) function build_model_from_ckpt(path, device=...) -> nn.Module
      C) function build_from_ckpt(path, device=...) -> nn.Module
    """
    dev = torch.device(device)

    # B/C: builder
    for fn_name in ["build_model_from_ckpt", "build_from_ckpt", "build_model"]:
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            m = fn(ckpt_path, device=device)
            if isinstance(m, tuple):
                m = m[0]
            if not isinstance(m, torch.nn.Module):
                raise RuntimeError(f"{fn_name} returned non-module: {type(m)}")
            m.to(dev).eval()
            return m

    # A: GraphSeg class
    Cls = getattr(mod, "GraphSeg", None)
    if Cls is None:
        raise RuntimeError("GraphSeg not found in model_py (and no build_* function found).")

    # Try init with ckpt_path kw
    try:
        m = Cls(ckpt_path=ckpt_path, device=device)
        if isinstance(m, tuple):
            m = m[0]
        if not isinstance(m, torch.nn.Module):
            raise RuntimeError(f"GraphSeg(ckpt_path=...) returned non-module: {type(m)}")
        m.to(dev).eval()
        return m
    except TypeError:
        pass

    # Try init without args then load_ckpt
    m = Cls()
    if isinstance(m, tuple):
        m = m[0]
    if not isinstance(m, torch.nn.Module):
        raise RuntimeError(f"GraphSeg() returned non-module: {type(m)}")
    load_fn = getattr(m, "load_ckpt", None)
    if callable(load_fn):
        load_fn(ckpt_path, device=device)
    else:
        # fallback: torch.load and load_state_dict if possible
        ck = torch.load(ckpt_path, map_location="cpu")
        sd = ck.get("state_dict", ck)
        if hasattr(m, "load_state_dict"):
            m.load_state_dict(sd, strict=False)
        else:
            raise RuntimeError("GraphSeg has no load_ckpt and no load_state_dict.")
    m.to(dev).eval()
    return m


@torch.no_grad()
def _forward_logits(model, x: torch.Tensor, dyn_on_eval: str | None, eval_dyn_k: int | None):
    # Try common signatures
    try:
        y = model(x, dyn_on_eval=dyn_on_eval, eval_dyn_k=eval_dyn_k)
    except TypeError:
        y = model(x)

    # Many models return (out, raw, graph) or (logits, ...)
    if isinstance(y, tuple) or isinstance(y, list):
        y0 = y[0]
    else:
        y0 = y
    return y0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_py", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--autocrop", type=int, default=1)
    ap.add_argument("--square", type=int, default=1)

    ap.add_argument("--clahe", type=int, default=0)
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_tile", type=int, default=8)

    ap.add_argument("--post", type=int, default=1)
    ap.add_argument("--save_in", type=int, default=0)

    ap.add_argument("--dyn_on_eval", type=str, default="feat")
    ap.add_argument("--eval_dyn_k", type=int, default=16)

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--swap12", type=int, default=0, help="Swap labels 1<->2 in predicted mask012 before saving.")

    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{ts()}][cfg] device={args.device}", flush=True)
    print(f"[{ts()}][cfg] in_path={in_path}", flush=True)
    print(f"[{ts()}][cfg] out_dir={out_dir}", flush=True)
    print(f"[{ts()}][cfg] img_size={args.img_size} swap12={int(args.swap12)}", flush=True)
    print(f"[{ts()}][cfg] autocrop={int(args.autocrop)} square={int(args.square)} clahe={int(args.clahe)} post={int(args.post)} save_in={int(args.save_in)}", flush=True)
    print(f"[{ts()}][cfg] dyn_on_eval={args.dyn_on_eval} eval_dyn_k={args.eval_dyn_k}", flush=True)

    files = _infer_file_list(in_path, limit=args.limit)
    if not files:
        raise SystemExit(f"[ERR] no images found under: {in_path}")

    print(f"[{ts()}][run] n_files={len(files)}", flush=True)
    for p in files[:5]:
        print(f"  - {p}", flush=True)

    mod = _import_model_module(args.model_py)
    model = _load_model(mod, args.ckpt, args.device)

    dev = torch.device(args.device)
    H = W = int(args.img_size)

    empty_like = 0
    nonbg_fracs = []
    mean_confs = []

    for i, fp in enumerate(files, start=1):
        stem = fp.stem
        img = _read_image_rgb(fp)

        crop_ok = 1
        crop_box = (0, 0, img.shape[1], img.shape[0])

        if args.autocrop:
            # For now: simple center crop (robust across datasets)
            img, crop_box = _center_square(img)
        if args.square and img.shape[0] != img.shape[1]:
            img, crop_box = _center_square(img)

        if args.clahe:
            img = _clahe_rgb(img, clip=args.clahe_clip, tile=args.clahe_tile)

        img_r = _resize(img, H, cv2.INTER_LINEAR)

        x = torch.from_numpy(img_r.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(dev)

        logits = _forward_logits(model, x, args.dyn_on_eval, args.eval_dyn_k)  # [1,C,H,W]
        if logits.ndim != 4:
            raise RuntimeError(f"Expected logits with shape [B,C,H,W], got {tuple(logits.shape)}")

        probs = torch.softmax(logits, dim=1).clamp(1e-9, 1.0)
        conf, pred = torch.max(probs, dim=1)  # [1,H,W]
        pred = pred[0].detach().cpu().numpy().astype(np.uint8)
        conf = conf[0].detach().cpu().numpy().astype(np.float32)

        if args.swap12:
            pred = _swap12_mask012(pred)

        # post (very light): keep labels in {0,1,2}
        if args.post:
            pred = pred.astype(np.uint8)

        counts = {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))}
        tot = int(pred.size)
        nonbg = tot - counts.get(0, 0)
        nonbg_frac = float(nonbg / max(1, tot))
        mean_conf = float(conf.mean())

        if nonbg == 0:
            empty_like += 1
        nonbg_fracs.append(nonbg_frac)
        mean_confs.append(mean_conf)

        # save
        cv2.imwrite(str(out_dir / f"{stem}__mask012.png"), pred)

        if args.save_in:
            cv2.imwrite(str(out_dir / f"{stem}__in.png"), cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))

        # overlay
        overlay = img_r.copy()
        col = np.zeros_like(overlay)
        col[pred == 1] = (255, 215, 0)   # ring
        col[pred == 2] = (255, 64, 64)   # cup
        overlay = cv2.addWeighted(overlay, 1.0, col, 0.45, 0.0)
        cv2.imwrite(str(out_dir / f"{stem}__overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # dbg
        with open(out_dir / f"{stem}__dbg.txt", "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "file": str(fp),
                "stem": stem,
                "crop_ok": int(crop_ok),
                "crop_box": list(map(int, crop_box)),
                "img_size": int(args.img_size),
                "swap12": int(args.swap12),
                "counts": counts,
                "nonbg_frac": nonbg_frac,
                "mean_conf": mean_conf,
            }, indent=2) + "\n")

        if (i == 1) or (i == len(files)) or (i % 200 == 0):
            print(f"[{ts()}][{i:04d}/{len(files):04d}] {fp.name} | crop_ok={crop_ok} nonbg_frac={nonbg_frac:.4f} mean_conf={mean_conf:.3f} counts={counts}", flush=True)

    def _stats(a):
        a = np.asarray(a, dtype=np.float64)
        return float(a.mean()), float(np.median(a)), float(a.min()), float(a.max())

    nb_m, nb_md, nb_min, nb_max = _stats(nonbg_fracs)
    mc_m, mc_md, mc_min, mc_max = _stats(mean_confs)

    print(f"[{ts()}][SUM] empty_like={empty_like}/{len(files)}", flush=True)
    print(f"[{ts()}][SUM] nonbg_frac mean={nb_m:.4f} median={nb_md:.4f} min={nb_min:.4f} max={nb_max:.4f}", flush=True)
    print(f"[{ts()}][SUM] mean_conf  mean={mc_m:.3f} median={mc_md:.3f} min={mc_min:.3f} max={mc_max:.3f}", flush=True)


if __name__ == "__main__":
    main()
