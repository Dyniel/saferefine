#!/usr/bin/env python3
import os, sys, re, csv, random
from pathlib import Path
import importlib.util

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F

def ts():
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def _strip_ext(x: str):
    x = str(x).strip()
    x = re.sub(r"\.(jpg|jpeg|png|bmp|tif|tiff)$", "", x, flags=re.IGNORECASE)
    return x

def load_builder(root: Path):
    cands = [
        root / "tools" / "builder.py",
        root / "builder.py",
    ]
    for p in cands:
        if p.exists():
            spec = importlib.util.spec_from_file_location("jaskra_builder", str(p))
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)
            if hasattr(mod, "build_model_from_args"):
                return mod
    raise FileNotFoundError(f"build_model_from_args not found. Looked for: {[str(c) for c in cands]}")

def sanitize_mask_to_012(mask_u8: np.ndarray) -> np.ndarray:
    m = mask_u8
    if m.ndim == 3 and m.shape[-1] == 3:
        # keep RGB as-is for color-uniques; but we will map by palette below
        pass
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    # If RGB, map colors -> 0/1/2 by area heuristic
    if m.ndim == 3 and m.shape[-1] == 3:
        rgb = m.reshape(-1, 3)
        cols, cnts = np.unique(rgb, axis=0, return_counts=True)
        bg_i = int(np.argmax(cnts))
        bg_col = cols[bg_i]
        others = [(tuple(cols[i].tolist()), int(cnts[i])) for i in range(len(cols)) if i != bg_i]
        if len(others) == 0:
            return np.zeros((m.shape[0], m.shape[1]), np.uint8)
        # sort remaining by area: disc larger than cup
        others.sort(key=lambda x: x[1], reverse=True)
        disc_col = np.array(others[0][0], dtype=np.uint8)
        cup_col  = np.array(others[-1][0], dtype=np.uint8) if len(others) >= 2 else None

        disc = np.all(m == disc_col[None,None,:], axis=-1)
        cup  = np.all(m == cup_col[None,None,:], axis=-1) if cup_col is not None else np.zeros_like(disc)

        ring = disc & (~cup)
        out = np.zeros((m.shape[0], m.shape[1]), np.uint8)
        out[ring] = 1
        out[cup]  = 2
        return out

    # else grayscale-like
    g = m.astype(np.int32)
    vals, cnts = np.unique(g, return_counts=True)
    bg_val = int(vals[int(np.argmax(cnts))])
    others = [int(v) for v in vals.tolist() if int(v) != bg_val]
    if len(others) == 0:
        return np.zeros_like(g, np.uint8)
    # disc bigger than cup -> pick disc as value with larger count among others
    areas = {v: int(cnts[list(vals).index(v)]) for v in others}
    disc_val = max(areas, key=areas.get)
    cup_val = min(areas, key=areas.get) if len(others) >= 2 else None

    disc = (g == disc_val)
    cup  = (g == cup_val) if cup_val is not None else np.zeros_like(disc)
    ring = disc & (~cup)

    out = np.zeros_like(g, np.uint8)
    out[ring] = 1
    out[cup]  = 2
    return out

def dice_bin(p: np.ndarray, g: np.ndarray) -> float:
    p = p.astype(np.uint8); g = g.astype(np.uint8)
    inter = int((p & g).sum())
    den = int(p.sum() + g.sum())
    if den == 0:
        return 1.0
    return float((2.0 * inter) / (den + 1e-6))

def compute_CpD(pred012: np.ndarray, gt012: np.ndarray) -> float:
    C = dice_bin(pred012 == 2, gt012 == 2)
    D = dice_bin(pred012 > 0,  gt012 > 0)
    return float(C + D)

def list_pairs(img_dir: Path, msk_dir: Path):
    def is_junk(p: Path):
        return p.name.startswith("._") or p.name.startswith(".DS_Store")

    imgs = [p for p in img_dir.glob("*") if p.is_file() and not is_junk(p)]
    msks = [p for p in msk_dir.glob("*") if p.is_file() and not is_junk(p)]

    img_map = {}
    for p in imgs:
        k = _strip_ext(p.name)
        img_map.setdefault(k, []).append(p)

    msk_map = {}
    for p in msks:
        k = _strip_ext(p.name)
        msk_map.setdefault(k, []).append(p)

    keys = sorted(set(img_map.keys()) & set(msk_map.keys()))
    pairs = []
    for k in keys:
        # if duplicates exist, prefer non-._ already filtered; if still multiple, pick first stable
        ip = sorted(img_map[k])[0]
        mp = sorted(msk_map[k])[0]
        pairs.append((k, ip, mp))
    return pairs

def read_pairs_csv(pairs_csv: Path):
    rows = list(csv.DictReader(pairs_csv.open("r", newline="")))
    # try common columns written by your audit
    for r in rows[:3]:
        pass
    out = []
    for r in rows:
        # accept several possible headers
        img = r.get("img_path") or r.get("img") or r.get("image") or r.get("img_file") or r.get("img_dir")
        msk = r.get("msk_path") or r.get("msk") or r.get("mask") or r.get("msk_file") or r.get("msk_dir")
        stem = r.get("stem") or r.get("id") or r.get("name") or r.get("key") or ""
        if img and msk:
            out.append((_strip_ext(stem) if stem else _strip_ext(Path(img).name), Path(img), Path(msk)))
    return out

def imread_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)

def mask_color(mask012: np.ndarray) -> np.ndarray:
    # 0=bg, 1=disc ring, 2=cup
    h, w = mask012.shape
    out = np.zeros((h, w, 3), np.uint8)
    out[mask012 == 1] = (255, 215, 0)   # gold
    out[mask012 == 2] = (255, 64, 64)   # red
    return out

def overlay(img_rgb: np.ndarray, mask012: np.ndarray, alpha=0.45) -> np.ndarray:
    col = mask_color(mask012)
    out = (img_rgb.astype(np.float32) * 1.0 + col.astype(np.float32) * alpha).clip(0,255).astype(np.uint8)
    return out

def resize_img_mask(img_rgb: np.ndarray, m012: np.ndarray, size: int):
    if img_rgb.shape[0] != size or img_rgb.shape[1] != size:
        img_rgb = np.array(Image.fromarray(img_rgb).resize((size, size), resample=Image.BILINEAR))
    if m012.shape[0] != size or m012.shape[1] != size:
        m012 = np.array(Image.fromarray(m012).resize((size, size), resample=Image.NEAREST))
    return img_rgb, m012

def to_tensor(img_rgb_u8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(img_rgb_u8).permute(2,0,1).float() / 255.0
    # ImageNet norm (bez tego często robi się “all-bg” na out-of-domain)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    x = (x - mean) / std
    return x

def panel_save(out_png: Path, img_rgb: np.ndarray, gt012: np.ndarray, pr012: np.ndarray, title: str):
    # layout: top row (img | gt | pred), bottom row (img+gt | img+pred | blank)
    a = img_rgb
    b = mask_color(gt012)
    c = mask_color(pr012)
    d = overlay(img_rgb, gt012)
    e = overlay(img_rgb, pr012)
    z = np.zeros_like(a)

    top = np.concatenate([a,b,c], axis=1)
    bot = np.concatenate([d,e,z], axis=1)
    panel = np.concatenate([top, bot], axis=0)

    im = Image.fromarray(panel)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = None
    dr.rectangle([0,0, panel.shape[1], 30], fill=(0,0,0))
    dr.text((8,4), title, fill=(255,255,255), font=font)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/home/student2/jaskra")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--pairs_csv", type=str, default="")
    ap.add_argument("--img_dir", type=str, default="")
    ap.add_argument("--msk_dir", type=str, default="")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.root)
    ckpt_p = Path(args.ckpt)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.pairs_csv:
        pairs = read_pairs_csv(Path(args.pairs_csv))
    else:
        assert args.img_dir and args.msk_dir, "Give --pairs_csv or (--img_dir and --msk_dir)"
        pairs = list_pairs(Path(args.img_dir), Path(args.msk_dir))

    if len(pairs) == 0:
        print("[FAIL] no pairs found", flush=True)
        sys.exit(2)

    # sample
    pairs = random.sample(pairs, k=min(args.n, len(pairs)))

    # build + load model from ckpt args
    builder = load_builder(root)
    ck = torch.load(str(ckpt_p), map_location="cpu")
    args_dict = ck.get("args", {}) or {}
    model = builder.build_model_from_args(args_dict)
    model.eval()
    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(dev)

    # strict-ish load
    sd = ck.get("state_dict", None) or ck.get("model", None) or ck.get("model_state_dict", None)
    if sd is None:
        # some checkpoints store directly
        if isinstance(ck, dict) and any(k.endswith("weight") for k in ck.keys()):
            sd = ck
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[cfg] loaded state_dict strict=False | missing={len(miss)} unexpected={len(unexp)}", flush=True)

    cps = []
    for i, (stem, img_p, msk_p) in enumerate(pairs, 1):
        img = imread_rgb(img_p)
        mraw = np.array(Image.open(msk_p))
        gt012 = sanitize_mask_to_012(mraw)

        img, gt012 = resize_img_mask(img, gt012, args.img_size)

        x = to_tensor(img).unsqueeze(0).to(dev)
        with torch.no_grad():
            logits = model(x)
            # support both “logits only” and “(out, raw, graph)” signatures
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pr = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        cp = compute_CpD(pr, gt012)
        cps.append(cp)

        title = f"{stem} | CpD={cp:.3f} | img={img_p.name} msk={msk_p.name}"
        out_png = outdir / f"{i:03d}_{stem}.png"
        panel_save(out_png, img, gt012, pr, title)

        if i <= 5:
            # quick sanity counts
            gt_vals, gt_cnt = np.unique(gt012, return_counts=True)
            pr_vals, pr_cnt = np.unique(pr, return_counts=True)
            gt_counts = {int(v): int(c) for v,c in zip(gt_vals, gt_cnt)}
            pr_counts = {int(v): int(c) for v,c in zip(pr_vals, pr_cnt)}
            print(f"[{i:03d}] {stem} CpD={cp:.3f} | gt_counts={gt_counts} pred_counts={pr_counts}", flush=True)

    cps = np.array(cps, np.float32)
    print(f"[OK] n={len(cps)} CpD mean={float(cps.mean()):.4f} median={float(np.median(cps)):.4f} min={float(cps.min()):.4f} max={float(cps.max()):.4f}", flush=True)
    print(f"[OK] panels -> {outdir}", flush=True)

if __name__ == "__main__":
    main()
