#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, json, math, argparse, importlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# optional
CRF_OK = False
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    CRF_OK = True
except Exception:
    CRF_OK = False

import cv2


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def dice_bin(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    inter = int((pred & gt).sum())
    den = int(pred.sum() + gt.sum())
    if den == 0:
        return 1.0
    return float((2 * inter) / (den + 1e-6))


def enforce_disc_cup(mask012):
    # mask: 0 bg, 1 ring, 2 cup
    disc = (mask012 > 0)
    cup = (mask012 == 2)
    # cup must be inside disc
    cup = cup & disc
    out = np.zeros_like(mask012, np.uint8)
    out[disc & (~cup)] = 1
    out[cup] = 2
    return out


def largest_cc(binmask):
    # keep largest connected component (8-connect)
    binmask = binmask.astype(np.uint8)
    n, lab = cv2.connectedComponents(binmask, connectivity=8)
    if n <= 1:
        return binmask
    # labels: 0 is bg
    areas = [(lab == i).sum() for i in range(1, n)]
    keep = 1 + int(np.argmax(areas))
    return (lab == keep).astype(np.uint8)


def fill_holes(binmask):
    # fill holes via floodfill from border
    m = binmask.astype(np.uint8)
    h, w = m.shape[:2]
    ff = m.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 1)
    inv = (ff == 0).astype(np.uint8)
    out = m | inv
    return out.astype(np.uint8)


def morph_refine(mask012, k=7, open_it=1, close_it=2):
    # OD/OC postprocess: CC, fill holes, smooth
    disc = (mask012 > 0).astype(np.uint8)
    cup  = (mask012 == 2).astype(np.uint8)

    disc = largest_cc(disc)
    cup  = largest_cc(cup)

    disc = fill_holes(disc)
    cup  = fill_holes(cup)

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    for _ in range(open_it):
        disc = cv2.morphologyEx(disc, cv2.MORPH_OPEN, ker)
        cup  = cv2.morphologyEx(cup,  cv2.MORPH_OPEN, ker)
    for _ in range(close_it):
        disc = cv2.morphologyEx(disc, cv2.MORPH_CLOSE, ker)
        cup  = cv2.morphologyEx(cup,  cv2.MORPH_CLOSE, ker)

    # enforce cup inside disc
    cup = cup & disc

    out = np.zeros_like(mask012, np.uint8)
    out[(disc == 1) & (cup == 0)] = 1
    out[cup == 1] = 2
    return out


def crf_refine(img_u8_rgb, prob, params):
    """
    img_u8_rgb: HxWx3 uint8
    prob: CxHxW float (softmax probs)
    params: dict with keys:
      - iters
      - sxy_gaussian, compat_gaussian
      - sxy_bilateral, srgb_bilateral, compat_bilateral
    """
    if not CRF_OK:
        raise RuntimeError("pydensecrf not available")

    C, H, W = prob.shape
    d = dcrf.DenseCRF2D(W, H, C)

    # unary from softmax expects shape (C, H*W)
    U = unary_from_softmax(prob.reshape(C, -1))
    d.setUnaryEnergy(U)

    sxy_g = params["sxy_gaussian"]
    compat_g = params["compat_gaussian"]
    d.addPairwiseGaussian(sxy=sxy_g, compat=compat_g)

    sxy_b = params["sxy_bilateral"]
    srgb_b = params["srgb_bilateral"]
    compat_b = params["compat_bilateral"]
    d.addPairwiseBilateral(sxy=sxy_b, srgb=srgb_b, rgbim=img_u8_rgb, compat=compat_b)

    Q = np.array(d.inference(int(params["iters"])), dtype=np.float32)  # C x (H*W)
    Q = Q.reshape(C, H, W)
    return Q


class TinyRefiner(nn.Module):
    # learned refiner on logits: (B,3,H,W)->(B,3,H,W)
    def __init__(self, ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, 3, 1, padding=0),
        )

    def forward(self, x):
        return x + self.net(x)  # residual


@torch.no_grad()
def eval_one_epoch(base_model, refiner, loader, device, mode, crf_params=None, morph_params=None):
    base_model.eval()
    if refiner is not None:
        refiner.eval()

    t0 = time.time()
    n = 0
    sumC = 0.0
    sumD = 0.0

    for xb, yb, meta in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        out = base_model(xb)[0]  # logits (B,3,H,W)
        if mode == "raw":
            logits = out
            pred = torch.argmax(logits, 1).detach().cpu().numpy()
        elif mode == "learned":
            logits = refiner(out)
            pred = torch.argmax(logits, 1).detach().cpu().numpy()
        elif mode == "morph":
            pred0 = torch.argmax(out, 1).detach().cpu().numpy()
            pred = []
            for i in range(pred0.shape[0]):
                m = pred0[i].astype(np.uint8)
                pred.append(morph_refine(m, **morph_params))
            pred = np.stack(pred, 0)
        elif mode == "crf":
            probs = F.softmax(out, dim=1).detach().cpu().numpy()  # B,C,H,W
            pred = []
            for i in range(probs.shape[0]):
                img = meta[i]["img_uint8"]  # HxWx3 uint8 RGB
                q = crf_refine(img, probs[i], crf_params)          # C,H,W
                m = np.argmax(q, axis=0).astype(np.uint8)
                m = enforce_disc_cup(m)
                pred.append(m)
            pred = np.stack(pred, 0)
        else:
            raise ValueError(mode)

        gt = yb.detach().cpu().numpy().astype(np.uint8)

        for i in range(pred.shape[0]):
            cup_p = (pred[i] == 2)
            cup_g = (gt[i] == 2)
            disc_p = (pred[i] > 0)
            disc_g = (gt[i] > 0)
            sumC += dice_bin(cup_p, cup_g)
            sumD += dice_bin(disc_p, disc_g)
            n += 1

    dt = time.time() - t0
    C = sumC / max(1, n)
    D = sumD / max(1, n)
    return {
        "n": int(n),
        "C": float(C),
        "D": float(D),
        "CpD": float(C + D),
        "time_min": float(dt / 60.0),
        "sec_per_img": float(dt / max(1, n)),
    }


def train_tiny_refiner(base_model, refiner, tr_loader, va_loader, device, epochs=5, lr=3e-4):
    base_model.eval()  # frozen; we learn only refiner
    for p in base_model.parameters():
        p.requires_grad = False

    refiner.train()
    opt = torch.optim.AdamW(refiner.parameters(), lr=lr, weight_decay=1e-4)

    best = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss_sum = 0.0
        n = 0

        for xb, yb, meta in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.no_grad():
                logits = base_model(xb)[0]  # B,3,H,W

            out = refiner(logits)
            loss = F.cross_entropy(out, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            opt.step()

            bs = xb.size(0)
            loss_sum += float(loss.detach().cpu()) * bs
            n += bs

        tr_loss = loss_sum / max(1, n)
        va = eval_one_epoch(base_model, refiner, va_loader, device, mode="learned")

        dt = (time.time() - t0) / 60.0
        score = va["CpD"]
        print(f"{ts()} [learned] ep={ep:02d} trCE={tr_loss:.4f}  val CpD={score:.6f} (C={va['C']:.6f} D={va['D']:.6f})  {dt:.2f} min", flush=True)

        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()}

    if best_state is not None:
        refiner.load_state_dict(best_state, strict=True)

    return best


def pick_ckpt(run_dir):
    p = Path(run_dir)
    if p.is_file() and p.suffix == ".pt":
        return p
    cands = list(p.rglob("best.pt"))
    if not cands:
        cands = list(p.rglob("*.pt"))
    if not cands:
        raise FileNotFoundError(f"No .pt found under: {p}")
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def build_base_model_from_ckpt(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    args = ck.get("args", {})
    # your builder
    import tools.model_builder_graphseg as b
    model = b.build_model_from_args(args)
    model = model.to(device)

    # IMPORTANT: handle lazy heads (feat_proj/seg_head) by dummy forward BEFORE strict load
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros((2, 3, int(args.get("img_size", 512)), int(args.get("img_size", 512))), device=device)
        _ = model(dummy)

    model.load_state_dict(ck["state_dict"], strict=True)
    return model, args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="run dir OR direct path to .pt")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--crf_grid", type=str, default="small", choices=["small","off"])
    ap.add_argument("--crf_pick", type=str, default="auto", help="auto or json string with params")
    ap.add_argument("--morph", type=str, default="k7", choices=["k5","k7","k9","off"])

    ap.add_argument("--learned", type=str, default="on", choices=["on","off"])
    ap.add_argument("--learned_epochs", type=int, default=5)
    ap.add_argument("--learned_lr", type=float, default=3e-4)

    ap.add_argument("--batch_val", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch_train", type=int, default=10)

    args = ap.parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ckpt = pick_ckpt(args.run_dir)
    print(f"[{ts()}] CKPT={ckpt}", flush=True)

    base_model, a = build_base_model_from_ckpt(ckpt, device)

    # import dataset/loader from unified training module to match preprocessing 1:1
    m = importlib.import_module("train_refuge2_npz_unified")
    Refuge2NPZ = getattr(m, "Refuge2NPZ")
    collate = getattr(m, "collate")
    read_idx_file = getattr(m, "read_idx_file")

    npz_all = a["npz_all"]
    idx_val = a["idx_val"]
    idx_tr  = a["idx_train"]

    idx_va = read_idx_file(idx_val)
    idx_trl = read_idx_file(idx_tr)

    va_ds = Refuge2NPZ(npz_all, idx_va, img_size=int(a.get("img_size", 512)), train=False, verbose=False)
    tr_ds = Refuge2NPZ(npz_all, idx_trl, img_size=int(a.get("img_size", 512)), train=True, verbose=False)

    va_loader = torch.utils.data.DataLoader(
        va_ds, batch_size=int(args.batch_val), shuffle=False, num_workers=int(args.workers),
        pin_memory=True, persistent_workers=(int(args.workers) > 0), drop_last=False, collate_fn=collate
    )
    tr_loader = torch.utils.data.DataLoader(
        tr_ds, batch_size=int(args.batch_train), shuffle=True, num_workers=int(args.workers),
        pin_memory=True, persistent_workers=(int(args.workers) > 0), drop_last=True, collate_fn=collate
    )

    # ---- RAW ----
    raw = eval_one_epoch(base_model, None, va_loader, device, mode="raw")
    print(f"[{ts()}] RAW   CpD={raw['CpD']:.6f}  C={raw['C']:.6f} D={raw['D']:.6f}  sec/img={raw['sec_per_img']:.4f}", flush=True)

    # ---- MORPH ----
    morph = None
    if args.morph != "off":
        if args.morph == "k5":
            morph_params = dict(k=5, open_it=1, close_it=2)
        elif args.morph == "k7":
            morph_params = dict(k=7, open_it=1, close_it=2)
        else:
            morph_params = dict(k=9, open_it=1, close_it=2)
        morph = eval_one_epoch(base_model, None, va_loader, device, mode="morph", morph_params=morph_params)
        print(f"[{ts()}] MORPH CpD={morph['CpD']:.6f}  C={morph['C']:.6f} D={morph['D']:.6f}  sec/img={morph['sec_per_img']:.4f}  params={morph_params}", flush=True)

    # ---- CRF ----
    crf_best = None
    if args.crf_grid != "off":
        if not CRF_OK:
            print(f"[{ts()}] CRF  SKIP (pydensecrf missing). Install: python -m pip install --user pydensecrf", flush=True)
        else:
            if args.crf_pick != "auto":
                crf_params = json.loads(args.crf_pick)
                crf_best = eval_one_epoch(base_model, None, va_loader, device, mode="crf", crf_params=crf_params)
                print(f"[{ts()}] CRF(FIX) CpD={crf_best['CpD']:.6f} C={crf_best['C']:.6f} D={crf_best['D']:.6f} sec/img={crf_best['sec_per_img']:.4f} params={crf_params}", flush=True)
            else:
                # narrow grid (val only)
                grid = []
                # keep it small on purpose
                for iters in [5, 10]:
                    for sxy_g in [1, 3]:
                        for compat_g in [2, 4]:
                            for sxy_b in [20, 40]:
                                for srgb_b in [5, 10]:
                                    for compat_b in [4, 8]:
                                        grid.append(dict(
                                            iters=iters,
                                            sxy_gaussian=sxy_g, compat_gaussian=compat_g,
                                            sxy_bilateral=sxy_b, srgb_bilateral=srgb_b, compat_bilateral=compat_b,
                                        ))
                best = (-1.0, None, None)
                for i, p in enumerate(grid, 1):
                    r = eval_one_epoch(base_model, None, va_loader, device, mode="crf", crf_params=p)
                    if r["CpD"] > best[0]:
                        best = (r["CpD"], p, r)
                    print(f"[{ts()}] CRF grid {i:03d}/{len(grid)} CpD={r['CpD']:.6f}  sec/img={r['sec_per_img']:.4f}", flush=True)
                crf_best = best[2]
                pbest = best[1]
                print(f"[{ts()}] CRF(BEST) CpD={crf_best['CpD']:.6f} C={crf_best['C']:.6f} D={crf_best['D']:.6f} sec/img={crf_best['sec_per_img']:.4f} params={pbest}", flush=True)

    # ---- LEARNED ----
    learned = None
    if args.learned == "on":
        ref = TinyRefiner(ch=16).to(device)
        print(f"[{ts()}] LEARNED train start epochs={args.learned_epochs} lr={args.learned_lr}", flush=True)
        _ = train_tiny_refiner(base_model, ref, tr_loader, va_loader, device, epochs=int(args.learned_epochs), lr=float(args.learned_lr))
        learned = eval_one_epoch(base_model, ref, va_loader, device, mode="learned")
        print(f"[{ts()}] LEARNED CpD={learned['CpD']:.6f} C={learned['C']:.6f} D={learned['D']:.6f} sec/img={learned['sec_per_img']:.4f}", flush=True)

    print("\n=== SUMMARY (VAL) ===")
    def line(name, r):
        if r is None:
            return
        print(f"{name:8s} CpD={r['CpD']:.6f}  C={r['C']:.6f}  D={r['D']:.6f}  sec/img={r['sec_per_img']:.4f}  time_min={r['time_min']:.2f}")

    line("RAW", raw)
    line("MORPH", morph)
    line("CRF", crf_best)
    line("LEARNED", learned)

    print("\n[tip] Jeśli chcesz uczciwie porównać z GRAFEM: odpal ten sam skrypt na ckpt grafowym (dyn/grid) i porównaj, ale CRF/MORPH/LEARNED licz na NO_GRAPH backbone.", flush=True)


if __name__ == "__main__":
    main()
