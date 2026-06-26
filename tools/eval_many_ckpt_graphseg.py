#!/usr/bin/env python3
import argparse, time
from pathlib import Path
import importlib
import numpy as np
import torch

def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def dice_bin(pred_bin, gt_bin):
    p = pred_bin.astype(np.uint8)
    g = gt_bin.astype(np.uint8)
    inter = int((p & g).sum())
    den = int(p.sum() + g.sum())
    if den == 0:
        return 1.0
    return float((2 * inter) / (den + 1e-6))

def find_run_dirs(root: Path):
    # run dir = subdir that contains ckpt/best.pt (or any *.pt under ckpt)
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        ckpt_dir = d / "ckpt"
        if ckpt_dir.is_dir():
            pts = list(ckpt_dir.glob("*.pt"))
            if pts:
                out.append(d)
    return out

def pick_ckpt(run_dir: Path):
    ckpt_dir = run_dir / "ckpt"
    best = ckpt_dir / "best.pt"
    if best.is_file():
        return best
    pts = sorted(ckpt_dir.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"no .pt in {ckpt_dir}")
    return pts[-1]

@torch.no_grad()
def eval_val(model, dl, device, dyn_on_eval="feat", eval_dyn_k=0):
    model.eval()
    sumC = 0.0
    sumD = 0.0
    n = 0
    for xb, yb, meta in dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        try:
            out = model(xb, dyn_on_eval=dyn_on_eval, eval_dyn_k=eval_dyn_k)
        except TypeError:
            out = model(xb)

        logits = out[0] if isinstance(out, (list, tuple)) else out
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        gt   = yb.detach().cpu().numpy()

        for b in range(pred.shape[0]):
            c = dice_bin(pred[b] == 2, gt[b] == 2)  # cup
            d = dice_bin(pred[b] > 0,  gt[b] > 0)   # disc
            sumC += c
            sumD += d
            n += 1
    C = sumC / max(1, n)
    D = sumD / max(1, n)
    return n, C, D, (C + D)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root runs dir (contains run subdirs)")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--out_tsv", default=None)
    ap.add_argument("--pattern", default=None, help="optional substring filter on run_dir name")
    ap.add_argument("--max_runs", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(str(root))

    device = torch.device("cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[{ts()}] root={root} device={device}")

    run_dirs = find_run_dirs(root)
    if args.pattern:
        run_dirs = [d for d in run_dirs if args.pattern in d.name]
    if args.max_runs and args.max_runs > 0:
        run_dirs = run_dirs[:args.max_runs]

    if not run_dirs:
        print(f"[{ts()}] no run dirs found under root")
        return

    # import builder + unified dataset utils once
    import tools.model_builder_graphseg as b
    unified = importlib.import_module("train_refuge2_npz_unified")
    Refuge2NPZ = getattr(unified, "Refuge2NPZ")
    collate = getattr(unified, "collate")
    read_idx_file = getattr(unified, "read_idx_file")
    from torch.utils.data import DataLoader

    rows = []
    for rd in run_dirs:
        ckpt_path = pick_ckpt(rd)
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt["state_dict"]
            a = ckpt.get("args", {})
            img_size = int(a.get("img_size", 512))

            # build model
            model = b.build_model_from_args(a).to(device)

            # dummy forward to materialize lazy heads
            x = torch.zeros((2, 3, img_size, img_size), device=device)
            _ = model(x)

            # strict load now
            model.load_state_dict(state, strict=True)

            # build val loader from ckpt args
            npz_all = Path(a["npz_all"])
            idx_val = Path(a["idx_val"])
            bs = int(min(6, max(1, int(a.get("batch", 10)) // 2)))
            workers = int(a.get("workers", 4))

            ds = Refuge2NPZ(str(npz_all), read_idx_file(str(idx_val)), img_size=img_size, train=False, verbose=False)
            dl = DataLoader(
                ds,
                batch_size=bs,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
                persistent_workers=(workers > 0),
                drop_last=False,
                collate_fn=collate,
            )

            dyn_on_eval = str(a.get("dyn_on_eval","feat"))
            eval_dyn_k  = int(a.get("eval_dyn_k", a.get("dyn_k", 0)))

            t0 = time.time()
            n, C, D, CD = eval_val(model, dl, device, dyn_on_eval=dyn_on_eval, eval_dyn_k=eval_dyn_k)
            dt_min = (time.time() - t0) / 60.0

            rows.append([
                str(root), rd.name, "OK",
                f"{CD:.6f}", f"{C:.6f}", f"{D:.6f}", str(n),
                dyn_on_eval, str(eval_dyn_k),
                f"{dt_min:.2f}",
                str(ckpt_path),
            ])
            print(f"[{ts()}] OK  {rd.name}  C+D={CD:.6f}  (C={C:.6f} D={D:.6f}) n={n}")

        except Exception as e:
            rows.append([
                str(root), rd.name, "FAIL",
                "-", "-", "-", "-", "-", "-", "-", str(ckpt_path),
            ])
            print(f"[{ts()}] FAIL {rd.name} err={repr(e)}")

    header = ["root","run","status","val_CpD","val_C","val_D","n","dyn_on_eval","eval_dyn_k","time_min","ckpt"]
    out_tsv = Path(args.out_tsv) if args.out_tsv else (root / "_eval_many_ckpt.tsv")

    with out_tsv.open("w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"[{ts()}] wrote: {out_tsv}")
    print(f"[{ts()}] tip:")
    print(f"  column -ts $'\\t' {out_tsv} | less -S")

if __name__ == "__main__":
    main()
