import os
import sys
import glob
import json
import traceback
import torch

def pick_ckpt(path_or_dir: str):
    p = path_or_dir
    if os.path.isdir(p):
        cands = []
        cands += glob.glob(os.path.join(p, "ckpt", "best.pt"))
        cands += glob.glob(os.path.join(p, "ckpt", "ep*.pt"))
        cands += glob.glob(os.path.join(p, "*.pt"))
        cands = sorted(list(dict.fromkeys(cands)))
        if not cands:
            raise FileNotFoundError(f"No .pt found under: {p}")
        # prefer best.pt, else newest by mtime
        best = [x for x in cands if x.endswith("/ckpt/best.pt")]
        if best:
            return best[0]
        cands.sort(key=lambda x: os.path.getmtime(x))
        return cands[-1]
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"Not found: {p}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=False, default="")
    ap.add_argument("--run_dir", type=str, required=False, default="")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--B", type=int, default=2)
    args = ap.parse_args()

    if not args.ckpt and not args.run_dir:
        print("[err] pass --ckpt PATH or --run_dir DIR", flush=True)
        sys.exit(2)

    target = args.ckpt or args.run_dir
    ckpt_path = pick_ckpt(target)

    dev = args.device
    if dev.startswith("cuda") and (not torch.cuda.is_available()):
        dev = "cpu"
    device = torch.device(dev)

    print(f"[ckpt] {ckpt_path}", flush=True)
    ck = torch.load(ckpt_path, map_location="cpu")

    # robustly find args + state_dict
    ck_args = ck.get("args", None)
    sd = ck.get("state_dict", None)
    if sd is None and isinstance(ck, dict):
        # sometimes saved as plain state_dict
        looks_like_sd = all(isinstance(k, str) for k in ck.keys())
        if looks_like_sd:
            sd = ck
    if sd is None:
        raise RuntimeError("Could not find state_dict in checkpoint (expected ckpt['state_dict']).")

    if ck_args is None:
        print("[warn] ckpt has no 'args' dict; builder will use defaults.", flush=True)
        ck_args = {}

    from tools.model_builder_graphseg import build_model_from_args
    model = build_model_from_args(ck_args)
    model.to(device)

    # load weights (strict=False to survive minor name diffs)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("[load] missing keys (first 30):", flush=True)
        for k in missing[:30]:
            print("  -", k, flush=True)
    if unexpected:
        print("[load] unexpected keys (first 30):", flush=True)
        for k in unexpected[:30]:
            print("  -", k, flush=True)

    model.eval()
    x = torch.randn(args.B, 3, args.H, args.W, device=device)

    with torch.no_grad():
        out = model(x)
    # handle tuple output
    if isinstance(out, (tuple, list)):
        shapes = []
        for i, t in enumerate(out):
            if torch.is_tensor(t):
                shapes.append((i, tuple(t.shape)))
            else:
                shapes.append((i, type(t).__name__))
        print("[fwd] output tuple/list:", shapes, flush=True)
    else:
        print("[fwd] output:", tuple(out.shape), flush=True)

    # print score fields if present
    print("[ok] model build+load+forward successful", flush=True)
    # print a compact args dump (helpful for paper reproducibility)
    try:
        print("[args] keys:", sorted(list(ck_args.keys()))[:40], flush=True)
    except Exception:
        pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FAIL]", repr(e), flush=True)
        traceback.print_exc()
        sys.exit(1)
