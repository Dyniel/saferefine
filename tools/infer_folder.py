#!/usr/bin/env python3
import os
import time
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F


def load_train_py(train_py: str):
    spec = importlib.util.spec_from_file_location("train_mod", str(train_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def to_rgb_u8(bgr_or_any):
    img = bgr_or_any
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def autocrop_fundus(rgb_u8, min_area_frac=0.10, pad_frac=0.06):
    h, w = rgb_u8.shape[:2]
    g = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 3.0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() > 127:
        th = cv2.bitwise_not(th)

    ker = max(3, int(round(min(h, w) * 0.01)) | 1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((ker, ker), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((ker, ker), np.uint8))

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return rgb_u8, False, (0, 0, w, h)

    areas = np.array([cv2.contourArea(c) for c in cnts], dtype=np.float32)
    k = int(np.argmax(areas))
    area = float(areas[k])
    if area < (min_area_frac * h * w):
        return rgb_u8, False, (0, 0, w, h)

    x, y, ww, hh = cv2.boundingRect(cnts[k])
    pad = int(round(pad_frac * max(ww, hh)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + ww + pad)
    y1 = min(h, y + hh + pad)
    crop = rgb_u8[y0:y1, x0:x1]
    if crop.size == 0:
        return rgb_u8, False, (0, 0, w, h)
    return crop, True, (x0, y0, x1, y1)


def square_pad_rgb(rgb_u8):
    h, w = rgb_u8.shape[:2]
    s = max(h, w)
    top = (s - h) // 2
    bot = s - h - top
    left = (s - w) // 2
    right = s - w - left
    out = cv2.copyMakeBorder(rgb_u8, top, bot, left, right, cv2.BORDER_REFLECT_101)
    return out


def square_pad_mask(m_u8):
    h, w = m_u8.shape[:2]
    s = max(h, w)
    top = (s - h) // 2
    bot = s - h - top
    left = (s - w) // 2
    right = s - w - left
    out = cv2.copyMakeBorder(m_u8, top, bot, left, right, cv2.BORDER_CONSTANT, value=0)
    return out


def resize_to_rgb(rgb_u8, size):
    if rgb_u8.shape[0] != size or rgb_u8.shape[1] != size:
        rgb_u8 = cv2.resize(rgb_u8, (size, size), interpolation=cv2.INTER_LINEAR)
    return rgb_u8


def resize_to_mask(m_u8, size):
    if m_u8.shape[0] != size or m_u8.shape[1] != size:
        m_u8 = cv2.resize(m_u8, (size, size), interpolation=cv2.INTER_NEAREST)
    return m_u8


def apply_clahe_rgb(rgb_u8, clip=2.0, tile=8):
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return out


def overlay_rgb(rgb_u8, mask012, a=0.45):
    col = rgb_u8.copy().astype(np.float32)
    cup = (mask012 == 1)
    disc = (mask012 == 2)
    lay = col.copy()
    lay[cup] = (1 - a) * lay[cup] + a * np.array([220, 30, 30], np.float32)
    lay[disc] = (1 - a) * lay[disc] + a * np.array([30, 220, 30], np.float32)
    return np.clip(lay, 0, 255).astype(np.uint8)


def preprocess_image(path, img_size, do_autocrop, do_square, do_clahe, clahe_clip, clahe_tile):
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    rgb = to_rgb_u8(bgr)
    if rgb is None:
        raise RuntimeError(f"Unreadable: {path}")

    crop_ok = False
    crop_box = (0, 0, rgb.shape[1], rgb.shape[0])

    if do_autocrop:
        rgb, crop_ok, crop_box = autocrop_fundus(rgb)

    if do_square:
        rgb = square_pad_rgb(rgb)

    if do_clahe:
        rgb = apply_clahe_rgb(rgb, clip=clahe_clip, tile=clahe_tile)

    rgb = resize_to_rgb(rgb, img_size)

    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    return {
        "path": str(path),
        "rgb_proc": rgb,
        "x": x,
        "crop_ok": crop_ok,
        "crop_box": crop_box,
    }


def read_mask_gray(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.dtype != np.uint8:
        m = np.clip(m, 0, 255).astype(np.uint8)
    return m


def align_raw_mask_to_proc(raw_u8, crop_ok, crop_box, img_size, do_autocrop, do_square):
    m = raw_u8
    if do_autocrop and crop_ok:
        x0, y0, x1, y1 = crop_box
        m = m[y0:y1, x0:x1]
    if do_square:
        m = square_pad_mask(m)
    m = resize_to_mask(m, img_size)
    return m


def decode_origa_g1020_raw_to_gt012_fixed(raw_aligned_u8):
    cup = (raw_aligned_u8 == 2)
    disc = (raw_aligned_u8 == 1) | (raw_aligned_u8 == 2)
    gt = np.zeros_like(raw_aligned_u8, np.uint8)
    gt[disc] = 2
    gt[cup] = 1
    return gt


def dice_nan(pred012, gt012, cls):
    p = (pred012 == cls)
    g = (gt012 == cls)
    ps = int(p.sum())
    gs = int(g.sum())
    if ps == 0 and gs == 0:
        return np.nan
    inter = int((p & g).sum())
    return (2.0 * inter) / float(ps + gs)


def find_images(img_dir: Path):
    exts = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".tif", ".tiff")
    paths = []
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix in exts:
            paths.append(p)
    return sorted(paths)


def find_mask_by_stem(mask_dir: Path, stem: str):
    for ext in (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"):
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def write_csv(path: Path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            def esc(v):
                s = "" if v is None else str(v)
                s = s.replace('"', '""')
                if ("," in s) or ("\n" in s):
                    return f'"{s}"'
                return s
            f.write(",".join(esc(r.get(c, "")) for c in cols) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["origa", "g1020"])
    ap.add_argument("--images", required=True, type=str)
    ap.add_argument("--masks", default=None, type=str)
    ap.add_argument("--model-py", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--outdir", required=True, type=str)

    ap.add_argument("--img-size", default=512, type=int)
    ap.add_argument("--device", default=None, type=str)

    ap.add_argument("--autocrop", action="store_true")
    ap.add_argument("--no-autocrop", action="store_true")
    ap.add_argument("--square", action="store_true")
    ap.add_argument("--no-square", action="store_true")

    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--clahe-clip", default=2.0, type=float)
    ap.add_argument("--clahe-tile", default=8, type=int)

    ap.add_argument("--require-both-classes", action="store_true")

    ap.add_argument("--save-pred-masks", action="store_true")
    ap.add_argument("--save-overlays", action="store_true")

    ap.add_argument("--force-graph-off", action="store_true")
    ap.add_argument("--force-alpha", default=None, type=float)
    ap.add_argument("--force-dyn-on", default=None, type=str)
    ap.add_argument("--force-dyn-k", default=None, type=int)

    ap.add_argument("--heartbeat", default=50, type=int)
    args = ap.parse_args()

    images_dir = Path(args.images)
    masks_dir = Path(args.masks) if args.masks else None
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    do_autocrop = True
    do_square = True

    if args.autocrop:
        do_autocrop = True
    if args.no_autocrop:
        do_autocrop = False

    if args.square:
        do_square = True
    if args.no_square:
        do_square = False

    if args.dataset == "g1020":
        do_clahe = True
    else:
        do_clahe = False

    if args.clahe:
        do_clahe = True
    if args.no_clahe:
        do_clahe = False

    train_mod = load_train_py(args.model_py)
    GraphSeg = getattr(train_mod, "GraphSeg")

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("state_dict", ck)
    ck_args = ck.get("args", {}) if isinstance(ck, dict) else {}

    backbone = ck_args.get("backbone", "deeplabv3_resnet50")
    feat_dim = int(ck_args.get("feat_dim", 256))
    hidden = int(ck_args.get("hidden", 512))
    depth = int(ck_args.get("depth", 3))
    graph_down = int(ck_args.get("graph_down", 2))
    grid4 = bool(ck_args.get("grid4", 1))
    dyn_on = str(ck_args.get("dyn_on", "feat"))
    dyn_k = int(ck_args.get("dyn_k", 4))
    dyn_window = int(ck_args.get("dyn_window", 2))
    alpha_graph = float(ck_args.get("alpha_graph", 0.55))

    dyn_on_eval = str(ck_args.get("dyn_on_eval", dyn_on))
    eval_dyn_k = int(ck_args.get("eval_dyn_k", dyn_k))
    alpha_eval = float(alpha_graph)

    if args.force_dyn_on is not None:
        dyn_on_eval = str(args.force_dyn_on)
    if args.force_dyn_k is not None:
        eval_dyn_k = int(args.force_dyn_k)
    if args.force_alpha is not None:
        alpha_eval = float(args.force_alpha)
    if args.force_graph_off:
        alpha_eval = 0.0

    model = GraphSeg(
        backbone=backbone,
        num_classes=3,
        feat_dim=feat_dim,
        hidden=hidden,
        depth=depth,
        graph_down=graph_down,
        grid4=grid4,
        dyn_on=dyn_on,
        dyn_k=dyn_k,
        dyn_window=dyn_window,
        alpha_graph=alpha_graph,
    ).to(device)

    with torch.no_grad():
        dummy = torch.zeros((1, 3, args.img_size, args.img_size), device=device, dtype=torch.float32)
        _ = model(dummy, dyn_on_eval=dyn_on_eval, eval_dyn_k=eval_dyn_k)

    model.load_state_dict(sd, strict=False)
    model.eval()

    img_paths = find_images(images_dir)

    pred_dir = outdir / "pred_masks"
    ovl_dir = outdir / "overlays"
    if args.save_pred_masks:
        pred_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        ovl_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    last = time.time()

    for i, img_path in enumerate(img_paths, 1):
        stem = img_path.stem

        try:
            sample = preprocess_image(
                img_path,
                img_size=args.img_size,
                do_autocrop=do_autocrop,
                do_square=do_square,
                do_clahe=do_clahe,
                clahe_clip=args.clahe_clip,
                clahe_tile=args.clahe_tile,
            )
        except Exception as e:
            rows.append({
                "stem": stem, "ok": 0, "reason": f"preprocess:{type(e).__name__}:{e}",
                "img": str(img_path), "mask": "",
            })
            continue

        x = sample["x"].to(device)
        with torch.no_grad():
            out, logits_up, node_up = model(x, dyn_on_eval=dyn_on_eval, eval_dyn_k=eval_dyn_k)
            if alpha_eval == 0.0:
                out_use = logits_up
            elif alpha_eval == 1.0:
                out_use = node_up if node_up is not None else out
            else:
                if node_up is None:
                    out_use = out
                else:
                    out_use = (1.0 - alpha_eval) * logits_up + alpha_eval * node_up

            prob = F.softmax(out_use, dim=1)[0].detach().cpu().numpy()
            pred = np.argmax(prob, axis=0).astype(np.uint8)
            conf = prob.max(axis=0)

        pr_bg = int((pred == 0).sum())
        pr_cup = int((pred == 1).sum())
        pr_disc = int((pred == 2).sum())
        pr_fg = int((pred > 0).sum())
        mean_conf = float(conf.mean())

        dice_cup = np.nan
        dice_disc = np.nan
        gt_cup_px = np.nan
        gt_disc_px = np.nan
        kept = 1
        reason = ""

        mask_path = ""
        if masks_dir is not None:
            mp = find_mask_by_stem(masks_dir, stem)
            if mp is None:
                kept = 0
                reason = "mask_missing"
            else:
                mask_path = str(mp)
                raw = read_mask_gray(mp)
                if raw is None:
                    kept = 0
                    reason = "gt_unreadable"
                else:
                    vals = set(np.unique(raw).tolist())
                    has1 = int(1 in vals)
                    has2 = int(2 in vals)
                    if args.require_both_classes and not (has1 and has2):
                        kept = 0
                        reason = "gt_not_both_classes"
                    else:
                        raw_al = align_raw_mask_to_proc(
                            raw,
                            crop_ok=sample["crop_ok"],
                            crop_box=sample["crop_box"],
                            img_size=args.img_size,
                            do_autocrop=do_autocrop,
                            do_square=do_square,
                        )
                        gt012 = decode_origa_g1020_raw_to_gt012_fixed(raw_al)
                        gt_cup_px = int((gt012 == 1).sum())
                        gt_disc_px = int((gt012 == 2).sum())
                        if args.require_both_classes and (gt_cup_px == 0 or gt_disc_px == 0):
                            kept = 0
                            reason = "gt_missing_after_align"
                        else:
                            dice_cup = float(dice_nan(pred, gt012, 1))
                            dice_disc = float(dice_nan(pred, gt012, 2))

        if args.save_pred_masks:
            out_p = pred_dir / f"{stem}.png"
            cv2.imwrite(str(out_p), pred.astype(np.uint8))

        if args.save_overlays:
            rgb = sample["rgb_proc"]
            ov = overlay_rgb(rgb, pred, a=0.45)
            out_o = ovl_dir / f"{stem}.jpg"
            cv2.imwrite(str(out_o), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

        rows.append({
            "stem": stem,
            "ok": 1,
            "kept": kept,
            "reason": reason,
            "img": str(img_path),
            "mask": mask_path,
            "dice_cup": dice_cup,
            "dice_disc": dice_disc,
            "gt_cup_px": gt_cup_px,
            "gt_disc_px": gt_disc_px,
            "pr_cup_px": pr_cup,
            "pr_disc_px": pr_disc,
            "pr_fg_px": pr_fg,
            "mean_conf": mean_conf,
            "crop_ok": int(sample["crop_ok"]),
            "crop_box": str(sample["crop_box"]),
        })

        now = time.time()
        if (args.heartbeat > 0 and (i % args.heartbeat == 0)) or (now - last > 12):
            kept_n = sum(1 for r in rows if r.get("kept", 1) == 1)
            print(f"[{i}/{len(img_paths)}] kept={kept_n} last={stem} pr_fg={pr_fg} mean_conf={mean_conf:.3f} dice_cup={dice_cup} dice_disc={dice_disc} elapsed={now-t0:.1f}s")
            last = now

    cols = [
        "stem", "ok", "kept", "reason",
        "dice_cup", "dice_disc",
        "gt_cup_px", "gt_disc_px",
        "pr_cup_px", "pr_disc_px", "pr_fg_px",
        "mean_conf", "crop_ok", "crop_box",
        "img", "mask",
    ]
    csv_path = outdir / f"{args.dataset}_infer.csv"
    write_csv(csv_path, rows, cols)
    print(f"WROTE {csv_path}")

    kept_rows = [r for r in rows if r.get("kept", 1) == 1]
    if masks_dir is not None:
        kept_eval = [r for r in kept_rows if isinstance(r.get("dice_cup", np.nan), float) or r.get("dice_cup") is not None]
        cup = np.array([r["dice_cup"] for r in kept_eval if r["dice_cup"] == r["dice_cup"]], dtype=np.float64)
        disc = np.array([r["dice_disc"] for r in kept_eval if r["dice_disc"] == r["dice_disc"]], dtype=np.float64)

        def summ(x):
            if x.size == 0:
                return {"n": 0}
            return {
                "n": int(x.size),
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "p10": float(np.percentile(x, 10)),
                "p90": float(np.percentile(x, 90)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            }

        print("SUMMARY kept-eval")
        print("cup :", summ(cup))
        print("disc:", summ(disc))


if __name__ == "__main__":
    main()
