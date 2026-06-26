#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, random, time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F

def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def die(msg, code=2):
    print(msg, flush=True)
    raise SystemExit(code)

def mmseg_norm_rgb_u8(img_rgb_u8):
    # mmseg default: mean/std in 0-255 space, bgr_to_rgb=True => we keep RGB
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    x = img_rgb_u8.astype(np.float32)
    x = (x - mean[None,None,:]) / std[None,None,:]
    return x

def read_pairs_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        # oczekujemy: stem,img,msk OR img_path,msk_path itp.
        for r in rd:
            rows.append(r)
    if not rows:
        die(f"[FAIL] empty pairs csv: {path}")
    return rows

def pick_paths(row):
    # wspieramy różne nazwy kolumn z audytu
    img = row.get("img") or row.get("img_path") or row.get("image") or row.get("image_path")
    msk = row.get("msk") or row.get("msk_path") or row.get("mask") or row.get("mask_path")
    stem = row.get("stem") or row.get("id") or row.get("name") or row.get("base") or ""
    if not img or not msk:
        return None
    return stem, img, msk

def load_refine_fair_eval_builder(root):
    root = Path(root)
    sys.path.insert(0, str(root))
    try:
        from tools import refine_fair_eval as rfe
    except Exception as e:
        die(f"[FAIL] cannot import tools.refine_fair_eval from {root}: {repr(e)}")
    if not hasattr(rfe, "builder"):
        die("[FAIL] tools.refine_fair_eval has no attribute: builder")
    if not hasattr(rfe.builder, "build_model_from_args"):
        die("[FAIL] tools.refine_fair_eval.builder has no build_model_from_args")
    need = ["strict_load_with_lazy_heads", "compute_C_D_from_mask"]
    for k in need:
        if not hasattr(rfe, k):
            die(f"[FAIL] tools.refine_fair_eval missing: {k}")
    return rfe

def safe_imread_rgb(path):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def safe_imread_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        # często maska jest RGB (kolory), sprowadź do 1 kanału
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m

def sanitize_mask_to_012(mask_u8):
    m = mask_u8
    m = m.astype(np.int32)

    uniq = np.unique(m)
    if np.array_equal(np.sort(uniq), np.array([0,1,2], dtype=np.int32)):
        return m.astype(np.uint8)

    # fallback: zmapuj 3 najczęstsze wartości do 0/1/2 (tło największe)
    vals, cnts = np.unique(m, return_counts=True)
    order = np.argsort(-cnts)
    bg = int(vals[order[0]])
    rest = [int(v) for v in vals[order[1:]]]
    out = np.zeros_like(m, np.uint8)
    if len(rest) >= 1:
        out[m == rest[0]] = 1
    if len(rest) >= 2:
        out[m == rest[1]] = 2
    # jeśli więcej niż 3 wartości, resztę wrzuć do 1 (ring) żeby nie znikało
    if len(rest) > 2:
        for v in rest[2:]:
            out[m == v] = 1
    return out

def resize_pair(img_rgb, m012, size):
    if img_rgb.shape[0] != size or img_rgb.shape[1] != size:
        img_rgb = cv2.resize(img_rgb, (size,size), interpolation=cv2.INTER_LINEAR)
    if m012.shape[0] != size or m012.shape[1] != size:
        m012 = cv2.resize(m012, (size,size), interpolation=cv2.INTER_NEAREST)
    return img_rgb, m012

def colorize_012(m012):
    # 0=tło czarne, 1=disc/ring żółty, 2=cup czerwony
    out = np.zeros((m012.shape[0], m012.shape[1], 3), np.uint8)
    out[m012 == 1] = (255, 215, 0)
    out[m012 == 2] = (255, 64, 64)
    return out

def overlay(img_rgb, m012, alpha=0.45):
    c = colorize_012(m012)
    return cv2.addWeighted(img_rgb, 1.0, c, alpha, 0)

def save_panel(png_path, img_rgb, gt012, pr012):
    gt_c = colorize_012(gt012)
    pr_c = colorize_012(pr012)
    ov_gt = overlay(img_rgb, gt012)
    ov_pr = overlay(img_rgb, pr012)

    # 2x3: img | GT(mask) | PR(mask) / img+GT | img+PR | diff (GT xor PR)
    diff = np.zeros_like(gt_c)
    diff[(gt012 > 0) & (pr012 == 0)] = (0, 0, 255)      # FN blue-ish? (actually red in RGB is (255,0,0); keep simple)
    diff[(gt012 == 0) & (pr012 > 0)] = (0, 255, 0)      # FP green
    diff[(gt012 > 0) & (pr012 > 0)] = (255, 255, 255)   # overlap white

    top = np.concatenate([img_rgb, gt_c, pr_c], axis=1)
    bot = np.concatenate([ov_gt, ov_pr, diff], axis=1)
    panel = np.concatenate([top, bot], axis=0)
    cv2.imwrite(str(png_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/home/student2/jaskra")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--pairs", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = Path(args.ckpt)
    pairs_path = Path(args.pairs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] root={args.root}", flush=True)
    print(f"[cfg] ckpt={ckpt_path}", flush=True)
    print(f"[cfg] pairs={pairs_path}", flush=True)
    print(f"[cfg] outdir={outdir}", flush=True)
    print(f"[cfg] n={args.n} img_size={args.img_size}", flush=True)

    rfe = load_refine_fair_eval_builder(args.root)

    ck = torch.load(str(ckpt_path), map_location="cpu")
    sd = ck.get("state_dict", ck)

    # args do budowy modelu: jak są w ckpt, bierzemy; jak nie ma, próbujemy z nazwy runa
    args_dict = ck.get("args", None)
    if args_dict is None:
        # minimalny fallback: wyciągnij backbone z nazwy ścieżki
        s = str(ckpt_path)
        bb = "deeplabv3_resnet50" if "deeplabv3_resnet50" in s else ("segformer_b0" if "segformer_b0" in s else None)
        if bb is None:
            die("[FAIL] ckpt has no args and backbone not inferrable from path")
        args_dict = {"backbone": bb}
    model = rfe.builder.build_model_from_args(args_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # loader z rfe (obsługuje lazy heads itp.)
    rfe.strict_load_with_lazy_heads(model, sd, device, H=args.img_size, W=args.img_size)

    rows = read_pairs_csv(pairs_path)
    items = []
    for r in rows:
        got = pick_paths(r)
        if got is None:
            continue
        stem, img, msk = got
        if str(Path(img).name).startswith("._") or str(Path(msk).name).startswith("._"):
            continue
        items.append((stem, img, msk))
    if not items:
        die(f"[FAIL] no usable pairs in: {pairs_path}")

    # losuj N
    random.shuffle(items)
    items = items[:min(args.n, len(items))]

    cps = []
    for i, (stem, img_p, msk_p) in enumerate(items, 1):
        img_rgb = safe_imread_rgb(img_p)
        m_raw = safe_imread_mask(msk_p)
        if img_rgb is None or m_raw is None:
            print(f"[skip] cannot read: {stem} img={img_p} msk={msk_p}", flush=True)
            continue

        gt012 = sanitize_mask_to_012(m_raw)
        img_rgb, gt012 = resize_pair(img_rgb, gt012, args.img_size)

        x = mmseg_norm_rgb_u8(img_rgb)  # HWC float32
        x = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            # model może zwracać tuple (out, logits_up, node_logits_up)
            if isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
            pr012 = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        C, D = rfe.compute_C_D_from_mask(pr012, gt012)
        CpD = float(C + D)
        cps.append(CpD)

        gt_counts = {int(k): int(v) for k, v in zip(*np.unique(gt012, return_counts=True))}
        pr_counts = {int(k): int(v) for k, v in zip(*np.unique(pr012, return_counts=True))}

        png = outdir / f"{i:03d}_{stem}_CpD{CpD:.3f}.png"
        save_panel(png, img_rgb, gt012, pr012)

        print(f"[{i:03d}] {stem} CpD={CpD:.3f} | gt={gt_counts} pred={pr_counts} | {png.name}", flush=True)

    if cps:
        cps = np.array(cps, dtype=np.float32)
        print(f"[OK] n={len(cps)} CpD mean={cps.mean():.4f} median={np.median(cps):.4f} min={cps.min():.4f} max={cps.max():.4f}", flush=True)
        print(f"[OK] panels -> {outdir}", flush=True)
    else:
        die("[FAIL] no successful samples")

if __name__ == "__main__":
    main()
