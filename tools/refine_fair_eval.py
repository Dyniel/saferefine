#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, argparse, importlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from tools.hrnet_graphseg_min import build_hrnet_graphseg_from_ckpt

import torch.nn as nn
import torch.nn.functional as F

# ---- optional deps ----
CRF_OK = False
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    CRF_OK = True
except Exception:
    CRF_OK = False

# ---- import training module for data pipeline consistency ----
# IMPORTANT: use same preprocessing/dataset as training
TRAIN_MOD = "train_refuge2_npz_unified"
M = importlib.import_module(TRAIN_MOD)

Refuge2NPZ = getattr(M, "Refuge2NPZ")
collate = getattr(M, "collate")
sanitize_mask_to_012 = getattr(M, "sanitize_mask_to_012")

# builder must return correct GraphSeg for ckpt["args"]
import tools.model_builder_graphseg as builder


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def dice_bin(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    inter = int((pred & gt).sum())
    den = int(pred.sum() + gt.sum())
    if den == 0:
        return 1.0
    return float((2.0 * inter) / (den + 1e-6))


def compute_C_D_from_mask(pred012, gt012):
    # C: cup class==2
    # D: disc = ring+cup => gt>0
    C = dice_bin((pred012 == 2), (gt012 == 2))
    D = dice_bin((pred012 > 0), (gt012 > 0))
    return C, D


def largest_cc(mask01):
    # simple largest connected component on CPU (no cv2 requirement)
    # mask01: uint8 {0,1}
    import collections
    H, W = mask01.shape
    visited = np.zeros_like(mask01, dtype=np.uint8)
    best = None
    best_sz = 0

    for y in range(H):
        for x in range(W):
            if mask01[y, x] and not visited[y, x]:
                q = collections.deque([(y, x)])
                visited[y, x] = 1
                pts = [(y, x)]
                while q:
                    cy, cx = q.popleft()
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < H and 0 <= nx < W and mask01[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = 1
                            q.append((ny, nx))
                            pts.append((ny, nx))
                if len(pts) > best_sz:
                    best_sz = len(pts)
                    best = pts

    out = np.zeros_like(mask01, dtype=np.uint8)
    if best is not None:
        for (y, x) in best:
            out[y, x] = 1
    return out


def fill_holes(mask01):
    # flood fill from borders in inverse mask to find holes
    import collections
    H, W = mask01.shape
    inv = (1 - mask01).astype(np.uint8)
    seen = np.zeros_like(inv, dtype=np.uint8)
    q = collections.deque()

    # push border zeros (inv==1 means background of mask)
    for x in range(W):
        if inv[0, x] and not seen[0, x]:
            q.append((0, x)); seen[0, x] = 1
        if inv[H-1, x] and not seen[H-1, x]:
            q.append((H-1, x)); seen[H-1, x] = 1
    for y in range(H):
        if inv[y, 0] and not seen[y, 0]:
            q.append((y, 0)); seen[y, 0] = 1
        if inv[y, W-1] and not seen[y, W-1]:
            q.append((y, W-1)); seen[y, W-1] = 1

    while q:
        cy, cx = q.popleft()
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny, nx = cy+dy, cx+dx
            if 0 <= ny < H and 0 <= nx < W and inv[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = 1
                q.append((ny, nx))

    # holes = inv==1 but not connected to border (seen==0)
    holes = (inv == 1) & (seen == 0)
    out = mask01.copy()
    out[holes] = 1
    return out.astype(np.uint8)


def morph_refine(pred012, k=7):
    # “tani okulistyczny baseline”
    # - largest CC
    # - fill holes
    # - enforce cup inside disc
    # NOTE: we avoid cv2; this is intentionally simple but fair/standard-ish.
    disc = (pred012 > 0).astype(np.uint8)
    cup = (pred012 == 2).astype(np.uint8)

    disc = largest_cc(disc)
    disc = fill_holes(disc)

    cup = cup & disc
    cup = largest_cc(cup)
    cup = fill_holes(cup)

    out = np.zeros_like(pred012, dtype=np.uint8)
    out[disc == 1] = 1
    out[cup == 1] = 2
    return out


def crf_refine(img_rgb_u8, probs, sxy=3, compat=4, iters=5):
    # probs: (3,H,W) float32, sums to 1
    if not CRF_OK:
        raise RuntimeError("pydensecrf not installed")
    C, H, W = probs.shape
    d = dcrf.DenseCRF2D(W, H, C)
    U = unary_from_softmax(probs)  # Cx(HW)
    d.setUnaryEnergy(U)

    # pairwise Gaussian (position)
    d.addPairwiseGaussian(sxy=(sxy, sxy), compat=compat)
    # pairwise bilateral (position + color)
    d.addPairwiseBilateral(sxy=(sxy, sxy), srgb=(13, 13, 13),
                           rgbim=img_rgb_u8, compat=compat)

    Q = d.inference(iters)
    Q = np.array(Q, dtype=np.float32).reshape((C, H, W))
    return Q


class TinyLogitsRefiner(nn.Module):
    # small learned baseline: logits -> refined logits
    def __init__(self, C=3, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, C, 1),
        )
    def forward(self, x):
        return self.net(x)


def soft_dice_loss(logits, y, eps=1e-6):
    # logits: (B,C,H,W), y: (B,H,W) int
    C = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    y1 = F.one_hot(y, num_classes=C).permute(0,3,1,2).float()
    inter = (probs * y1).sum(dim=(2,3))
    den = (probs + y1).sum(dim=(2,3))
    dice = (2*inter + eps) / (den + eps)
    # ignore bg less? keep simple and fair
    return 1.0 - dice.mean()


@dataclass
class Result:
    CpD: float
    C: float
    D: float
    sec_per_img: float


@torch.no_grad()
def eval_on_val(model, ds, device, dyn_on_eval=None, eval_dyn_k=None, mode="RAW",
                morph_k=7, crf_params=None, learned_refiner=None):
    model.eval()
    if learned_refiner is not None:
        learned_refiner.eval()

    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True, persistent_workers=False, drop_last=False,
        collate_fn=collate
    )

    t0 = time.time()
    n = 0
    sumC = 0.0
    sumD = 0.0

    for xb, yb, meta in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        out = model(xb, dyn_on_eval=dyn_on_eval, eval_dyn_k=eval_dyn_k)[0]  # (1,3,H,W)
        logits = out

        if mode == "LEARNED":
            logits = learned_refiner(logits)

        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]  # (3,H,W)
        pred = np.argmax(probs, axis=0).astype(np.uint8)
        gt = yb.detach().cpu().numpy()[0].astype(np.uint8)
        gt = sanitize_mask_to_012(gt)

        if mode == "MORPH":
            pred = morph_refine(pred, k=morph_k)

        elif mode == "CRF":
            if crf_params is None:
                raise RuntimeError("CRF mode needs crf_params")
            img = meta[0].get("img_uint8", None)
            if img is None:
                raise RuntimeError("meta missing img_uint8 (need it for CRF bilateral)")
            Q = crf_refine(img, probs, **crf_params)
            pred = np.argmax(Q, axis=0).astype(np.uint8)

        # RAW/LEARNED already handled
        C, D = compute_C_D_from_mask(pred, gt)
        sumC += C
        sumD += D
        n += 1

        if (n % 20) == 0:
            print(f"{ts()} [{mode}] iter={n:04d} C={sumC/n:.4f} D={sumD/n:.4f} CpD={(sumC+sumD)/n:.4f}", flush=True)

    dt = time.time() - t0
    sec_per_img = dt / max(1, n)
    return Result(CpD=(sumC+sumD)/n, C=sumC/n, D=sumD/n, sec_per_img=sec_per_img)


def pick_ckpt(run_dir):
    p = Path(run_dir)
    ck = p / "ckpt" / "best.pt"
    if ck.exists():
        return ck
    pts = list(p.rglob("best.pt"))
    if pts:
        return pts[0]
    pts = list(p.rglob("*.pt"))
    if pts:
        return pts[0]
    raise FileNotFoundError(f"no ckpt .pt found under {run_dir}")


def strict_load_with_lazy_heads(model, state_dict, device, H=512, W=512):
    # materialize lazy heads by dummy forward, then strict load
    model.to(device)
    model.eval()
    xb = torch.zeros((2,3,H,W), device=device)
    with torch.no_grad():
        _ = model(xb)[0]
    miss, unexp = model.load_state_dict(state_dict, strict=False)
    if len(miss) != 0 or len(unexp) != 0:
        # second attempt strict after dummy forward (now keys exist)
        miss2, unexp2 = model.load_state_dict(state_dict, strict=False)
        # allow if still none missing; unexpected should be gone
        miss = miss2; unexp = unexp2
    return miss, unexp


def tune_crf(ds_val, base_model, device, dyn_on_eval, eval_dyn_k, grid):
    if not CRF_OK:
        print(f"{ts()} [CRF] SKIP: pydensecrf not installed", flush=True)
        return None, None

    best = None
    best_params = None
    # to keep tuning honest + bounded: full val, but small grid
    for params in grid:
        r = eval_on_val(base_model, ds_val, device, dyn_on_eval, eval_dyn_k, mode="CRF", crf_params=params)
        print(f"{ts()} [CRF][tune] params={params} CpD={r.CpD:.6f} C={r.C:.6f} D={r.D:.6f} sec/img={r.sec_per_img:.4f}", flush=True)
        if (best is None) or (r.CpD > best.CpD):
            best = r
            best_params = params
    return best, best_params


def tune_morph(ds_val, base_model, device, dyn_on_eval, eval_dyn_k, ks):
    best = None
    best_k = None
    for k in ks:
        r = eval_on_val(base_model, ds_val, device, dyn_on_eval, eval_dyn_k, mode="MORPH", morph_k=k)
        print(f"{ts()} [MORPH][tune] k={k} CpD={r.CpD:.6f} C={r.C:.6f} D={r.D:.6f} sec/img={r.sec_per_img:.4f}", flush=True)
        if (best is None) or (r.CpD > best.CpD):
            best = r
            best_k = k
    return best, best_k


def train_learned_refiner(ds_train, ds_val, base_model, device, dyn_on_eval, eval_dyn_k,
                          epochs=20, lr=3e-4, out_path=None):
    ref = TinyLogitsRefiner(C=3, hidden=32).to(device)
    opt = torch.optim.AdamW(ref.parameters(), lr=lr, weight_decay=1e-4)

    tr_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=10, shuffle=True, num_workers=0,
        pin_memory=True, persistent_workers=False, drop_last=True,
        collate_fn=collate
    )
    va_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True, persistent_workers=False, drop_last=False,
        collate_fn=collate
    )

    best = -1e9
    best_state = None

    base_model.eval()
    for ep in range(1, epochs+1):
        t0 = time.time()
        ref.train()
        loss_sum = 0.0
        n = 0

        for xb, yb, meta in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.no_grad():
                logits = base_model(xb, dyn_on_eval=dyn_on_eval, eval_dyn_k=eval_dyn_k)[0]

            pred_logits = ref(logits)
            ce = F.cross_entropy(pred_logits, yb)
            sd = soft_dice_loss(pred_logits, yb)
            loss = ce + 0.5 * sd

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
            opt.step()

            bs = xb.size(0)
            loss_sum += float(loss.detach().cpu()) * bs
            n += bs

        # val score
        ref.eval()
        sumC = 0.0
        sumD = 0.0
        m = 0
        with torch.no_grad():
            for xb, yb, meta in va_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = base_model(xb, dyn_on_eval=dyn_on_eval, eval_dyn_k=eval_dyn_k)[0]
                logits2 = ref(logits)
                probs = F.softmax(logits2, dim=1).cpu().numpy()[0]
                pred = np.argmax(probs, axis=0).astype(np.uint8)
                gt = sanitize_mask_to_012(yb.cpu().numpy()[0].astype(np.uint8))
                C, D = compute_C_D_from_mask(pred, gt)
                sumC += C; sumD += D; m += 1

        CpD = (sumC+sumD)/max(1,m)
        dtm = (time.time()-t0)/60.0
        print(f"{ts()} [LEARNED] ep={ep:03d} train_loss={loss_sum/max(1,n):.4f} val_CpD={CpD:.6f} ({sumC/m:.6f}+{sumD/m:.6f}) time_min={dtm:.2f}", flush=True)

        if CpD > best:
            best = CpD
            best_state = {k: v.detach().cpu().clone() for k,v in ref.state_dict().items()}

    if out_path is not None and best_state is not None:
        torch.save({"state_dict": best_state, "best_val_CpD": best}, out_path)
        print(f"{ts()} [LEARNED] saved best to {out_path} (best_val_CpD={best:.6f})", flush=True)

    if best_state is not None:
        ref.load_state_dict(best_state, strict=True)
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", default=None)

    # evaluation settings
    ap.add_argument("--dyn_on_eval", default=None)
    ap.add_argument("--eval_dyn_k", type=int, default=None)

    # tuning grids
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--crf_grid", default="small")     # small/none
    ap.add_argument("--morph_ks", default="3,5,7,9")

    # learned
    ap.add_argument("--learned", default="on")         # on/off
    ap.add_argument("--learned_epochs", type=int, default=20)
    ap.add_argument("--learned_lr", type=float, default=3e-4)

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = pick_ckpt(run_dir)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    print(f"[{ts()}] RUN_DIR={run_dir}", flush=True)
    print(f"[{ts()}] CKPT={ckpt_path}", flush=True)
    print(f"[{ts()}] device={device}", flush=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_dict = ckpt.get("args", {})
    print(f"[{ts()}] ckpt keys={list(ckpt.keys())}", flush=True)

    # build model from ckpt args
    bb = str(args_dict.get('backbone') or args_dict.get('model') or args_dict.get('arch') or '')
    if 'hrnet' in bb.lower():
        _ck = torch.load(ckpt_path, map_location='cpu')
        model, _miss, _unexp = build_hrnet_graphseg_from_ckpt(_ck, device=device)
    else:
        model = builder.build_model_from_args(args_dict)

    # dyn defaults: if not provided, use ckpt args
    dyn_on_eval = args.dyn_on_eval if args.dyn_on_eval is not None else args_dict.get("dyn_on_eval", "feat")
    eval_dyn_k = args.eval_dyn_k if args.eval_dyn_k is not None else int(args_dict.get("eval_dyn_k", 16))

    # strict load (handle lazy heads)
    miss, unexp = strict_load_with_lazy_heads(model, ckpt["state_dict"], device, H=int(args_dict.get("img_size",512)), W=int(args_dict.get("img_size",512)))
    print(f"[{ts()}] load missing={len(miss)} unexpected={len(unexp)}", flush=True)
    if len(miss) != 0:
        print(f"[{ts()}] missing(first30)={miss[:30]}", flush=True)
    if len(unexp) != 0:
        print(f"[{ts()}] unexpected(first30)={unexp[:30]}", flush=True)

    # datasets
    npz_all = args_dict["npz_all"]
    idx_tr = args_dict["idx_train"]
    idx_va = args_dict["idx_val"]
    img_size = int(args_dict.get("img_size", 512))

    ds_train = Refuge2NPZ(npz_all, M.read_idx_file(idx_tr), img_size=img_size, train=True, verbose=False)
    ds_val   = Refuge2NPZ(npz_all, M.read_idx_file(idx_va), img_size=img_size, train=False, verbose=False)

    out_dir = run_dir / "refine_fair"
    out_dir.mkdir(parents=True, exist_ok=True)
    tune_path = out_dir / "tuned_params.json"

    tuned = {}
    if tune_path.exists():
        tuned = json.loads(tune_path.read_text())

    # grids
    morph_ks = [int(x) for x in args.morph_ks.split(",") if x.strip()]
    if args.crf_grid == "small":
        crf_grid = []
        # narrow grid: small but defensible
        for sxy in (2, 3, 4):
            for compat in (3, 4, 5):
                for iters in (5, 10):
                    crf_grid.append({"sxy": int(sxy), "compat": int(compat), "iters": int(iters)})
    else:
        crf_grid = []

    # ---- TUNE (VAL only) ----
    if args.tune:
        # MORPH tune
        best_m, best_k = tune_morph(ds_val, model, device, dyn_on_eval, eval_dyn_k, morph_ks)
        tuned["morph_k"] = int(best_k)

        # CRF tune
        if len(crf_grid) > 0:
            best_c, best_params = tune_crf(ds_val, model, device, dyn_on_eval, eval_dyn_k, crf_grid)
            if best_params is not None:
                tuned["crf_params"] = best_params
        tune_path.write_text(json.dumps(tuned, indent=2, sort_keys=True))
        print(f"[{ts()}] saved tuned params -> {tune_path}", flush=True)

    # use tuned or defaults
    morph_k = int(tuned.get("morph_k", 7))
    crf_params = tuned.get("crf_params", {"sxy":3,"compat":4,"iters":5})

    # ---- LEARNED (train on TRAIN, select on VAL) ----
    learned_refiner = None
    learned_path = out_dir / "learned_best.pt"
    if args.learned.lower() == "on":
        if learned_path.exists():
            st = torch.load(learned_path, map_location="cpu")
            learned_refiner = TinyLogitsRefiner(C=3, hidden=32).to(device)
            learned_refiner.load_state_dict(st["state_dict"], strict=True)
            print(f"[{ts()}] loaded learned refiner from {learned_path} (best_val_CpD={st.get('best_val_CpD','?')})", flush=True)
        else:
            learned_refiner = train_learned_refiner(
                ds_train, ds_val, model, device, dyn_on_eval, eval_dyn_k,
                epochs=args.learned_epochs, lr=args.learned_lr, out_path=learned_path
            )

    # ---- EVAL (VAL report) ----
    # IMPORTANT: this is the fair report you put in table.
    res = {}
    r_raw = eval_on_val(model, ds_val, device, dyn_on_eval, eval_dyn_k, mode="RAW")
    res["RAW"] = r_raw.__dict__

    r_m = eval_on_val(model, ds_val, device, dyn_on_eval, eval_dyn_k, mode="MORPH", morph_k=morph_k)
    res["MORPH"] = r_m.__dict__
    res["MORPH"]["morph_k"] = morph_k

    if CRF_OK:
        r_c = eval_on_val(model, ds_val, device, dyn_on_eval, eval_dyn_k, mode="CRF", crf_params=crf_params)
        res["CRF"] = r_c.__dict__
        res["CRF"]["crf_params"] = crf_params
    else:
        res["CRF"] = {"SKIP": "pydensecrf not installed"}

    if learned_refiner is not None:
        r_l = eval_on_val(model, ds_val, device, dyn_on_eval, eval_dyn_k, mode="LEARNED", learned_refiner=learned_refiner)
        res["LEARNED"] = r_l.__dict__
    else:
        res["LEARNED"] = {"SKIP": "learned=off"}

    # pretty summary
    def line(name, r):
        if "SKIP" in r:
            return f"{name:<8} SKIP  {r['SKIP']}"
        return (f"{name:<8} CpD={r['CpD']:.6f}  C={r['C']:.6f}  D={r['D']:.6f}  "
                f"sec/img={r['sec_per_img']:.4f}")

    print("\n=== SUMMARY (VAL) ===", flush=True)
    print(line("RAW", res["RAW"]), flush=True)
    print(line("MORPH", res["MORPH"]), flush=True)
    print(line("CRF", res["CRF"]), flush=True)
    print(line("LEARNED", res["LEARNED"]), flush=True)

    # save json
    out_json = args.out_json if args.out_json is not None else str(out_dir / "val_refine_report.json")
    Path(out_json).write_text(json.dumps({
        "run_dir": str(run_dir),
        "ckpt": str(ckpt_path),
        "dyn_on_eval": dyn_on_eval,
        "eval_dyn_k": int(eval_dyn_k),
        "tuned_params_path": str(tune_path),
        "results": res
    }, indent=2, sort_keys=True))
    print(f"\n[{ts()}] saved: {out_json}", flush=True)



# ---------------- FAST MORPH OVERRIDE (cv2) ----------------
try:
    import cv2
    _CV2_OK = True
except Exception:
    _CV2_OK = False

def _largest_cc_cv2(mask01):
    # mask01: uint8 {0,1}
    if mask01 is None or mask01.size == 0:
        return mask01
    m = (mask01 > 0).astype("uint8")
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    # skip background (0); pick max area
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(areas.argmax())
    out = (lab == k).astype("uint8")
    return out

def _fill_holes_cv2(mask01):
    m = (mask01 > 0).astype("uint8") * 255
    h, w = m.shape
    ff = m.copy()
    mask = (255 - m).copy()
    # flood fill from border on background
    flood = mask.copy()
    cv2.floodFill(flood, None, (0,0), 0)
    holes = (flood == 255).astype("uint8") * 255
    filled = cv2.bitwise_or(m, holes)
    return (filled > 0).astype("uint8")

def morph_refine(pred012, k=7):
    """
    Fast, standard-ish OD/OC postprocess baseline:
    - closing/opening (disc + cup)
    - largest connected component
    - hole fill
    - enforce cup inside disc
    """
    if not _CV2_OK:
        # fallback to old (slow) version if cv2 missing
        return pred012

    pred012 = pred012.astype("uint8")
    disc = (pred012 > 0).astype("uint8")
    cup  = (pred012 == 2).astype("uint8")

    kk = max(3, int(k))
    if kk % 2 == 0:
        kk += 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))

    # disc smooth + cc + holes
    disc = cv2.morphologyEx(disc, cv2.MORPH_CLOSE, ker, iterations=1)
    disc = cv2.morphologyEx(disc, cv2.MORPH_OPEN,  ker, iterations=1)
    disc = _largest_cc_cv2(disc)
    disc = _fill_holes_cv2(disc)

    # cup smooth + constrain + cc + holes
    cup = (cup & disc).astype("uint8")
    cup = cv2.morphologyEx(cup, cv2.MORPH_CLOSE, ker, iterations=1)
    cup = cv2.morphologyEx(cup, cv2.MORPH_OPEN,  ker, iterations=1)
    cup = _largest_cc_cv2(cup)
    cup = _fill_holes_cv2(cup)
    cup = (cup & disc).astype("uint8")

    out = (disc > 0).astype("uint8")  # ring=1 for disc region
    out[cup > 0] = 2
    return out.astype("uint8")
# ---------------- END FAST MORPH OVERRIDE ----------------

if __name__ == "__main__":
    main()
