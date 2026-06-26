#!/usr/bin/env python3
import argparse, random
from pathlib import Path

def read_idx(p):
    lines = Path(p).read_text().splitlines()
    out=[]
    for ln in lines:
        ln=ln.strip()
        if not ln or ln.startswith("#"): 
            continue
        out.append(ln)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx_train", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fracs", default="0.25,0.50,0.75")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = read_idx(args.idx_train)
    n = len(base)
    rnd = random.Random(args.seed)
    perm = base[:]
    rnd.shuffle(perm)

    fracs = [float(x.strip()) for x in args.fracs.split(",") if x.strip()]
    for f in fracs:
        k = max(1, int(round(n*f)))
        sub = perm[:k]
        p = out_dir / f"idx_train_{int(round(f*100)):02d}pct_seed{args.seed}.txt"
        p.write_text("\n".join(sub) + "\n")
        print(f"[OK] {p} n={k}/{n} frac={f} seed={args.seed}")

if __name__ == "__main__":
    main()
