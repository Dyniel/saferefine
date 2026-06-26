import os, re, math, glob
from pathlib import Path
import argparse
import statistics as stats

FRAC_RE = re.compile(r'(?:^|[^0-9])(?:frac|pct|percent|p)?\s*(25|50|75)(?:[^0-9]|$)', re.IGNORECASE)
SEED_RE = re.compile(r'(?:^|_)s(\d+)(?:_|$)')
# metric patterns (order matters)
PATS = [
    re.compile(r'best\(C\+D\)\s*[:=]\s*([0-9]*\.[0-9]+|[0-9]+(?:\.[0-9]+)?)'),
    re.compile(r'best\(C\+D\)\s+([0-9]*\.[0-9]+|[0-9]+(?:\.[0-9]+)?)'),
    re.compile(r'\bCpD\s*=\s*([0-9]*\.[0-9]+|[0-9]+(?:\.[0-9]+)?)'),
    re.compile(r'Dice\(C\)\+Dice\(D\)\s*[:=]\s*([0-9]*\.[0-9]+|[0-9]+(?:\.[0-9]+)?)'),
]

def find_frac(name: str):
    # Tokenize by common separators and look for standalone 25/50/75.
    # Works for patterns like: "..._s1_25", "..._25_s1", "...-50", etc.
    s = name.replace("-", "_").lower()
    toks = [t for t in s.split("_") if t]
    fracs = []
    for t in toks:
        if t in ("25", "50", "75"):
            fracs.append(int(t))
        # also allow "p25" / "pct25" / "frac25" forms
        m = re.match(r'^(?:p|pct|frac|percent)?(25|50|75)$', t)
        if m:
            fracs.append(int(m.group(1)))
    return fracs[-1] if fracs else None


def find_seed(name: str):
    m = SEED_RE.search(name)
    return int(m.group(1)) if m else None

def find_method(name: str):
    n = name.lower()
    if "no_graph" in n:
        return "No-Graph"
    if "grid_only" in n:
        return "Grid-Only"
    if "dynk16" in n or "dyn_k16" in n or "dynk_16" in n or "dyn-k16" in n:
        return "Dyn-k=16"
    return None

def tail_read(path: Path, max_lines=4000):
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            # read last ~2MB max
            chunk = 2_000_000
            start = max(0, end - chunk)
            f.seek(start)
            data = f.read().decode("utf-8", errors="ignore")
        lines = data.splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception:
        return ""

def extract_metric_from_log(log_path: Path):
    txt = tail_read(log_path)
    if not txt:
        return None
    # search from bottom up by scanning reversed lines first (faster "last best")
    lines = txt.splitlines()
    for line in reversed(lines):
        for pat in PATS:
            m = pat.search(line)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass
    # fallback: global search
    for pat in PATS:
        m = pat.search(txt)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None

def candidate_logs(run_dir: Path):
    cands = []
    # common locations
    for pat in ["*.log", "logs/*.log", "log*.log", "**/*.log"]:
        for p in run_dir.glob(pat):
            if p.is_file():
                cands.append(p)
    # prefer shorter paths (top-level logs) + newest
    cands = list({p.resolve() for p in cands})
    cands.sort(key=lambda p: (len(str(p)), -p.stat().st_mtime))
    return cands

def scan_runs(root: Path, max_depth=6):
    runs = []
    root = root.resolve()
    # heuristic: any dir containing ckpt/best.pt OR logs/*.log
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        try:
            rel_depth = len(d.relative_to(root).parts)
        except Exception:
            continue
        if rel_depth > max_depth:
            continue
        has_ckpt = (d / "ckpt" / "best.pt").is_file()
        has_logs = any((d / "logs").glob("*.log")) or any(d.glob("*.log"))
        if has_ckpt or has_logs:
            runs.append(d)
    # de-dup: keep deepest unique run dirs by name
    uniq = {}
    for d in runs:
        uniq[str(d.resolve())] = d.resolve()
    return list(uniq.values())

def mean_std(xs):
    xs = list(xs)
    if len(xs) == 0:
        return (math.nan, math.nan)
    if len(xs) == 1:
        return (xs[0], 0.0)
    return (stats.mean(xs), stats.pstdev(xs) if len(xs) > 1 else 0.0)

def fmt_pm(mu, sd, nd=4):
    if math.isnan(mu):
        return "XX"
    return f"{mu:.{nd}f} $\\pm$ {sd:.{nd}f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="/home/student2/jaskra/runs")
    ap.add_argument("--out_tex", default="/home/student2/jaskra/runs/_small_data_table.tex")
    ap.add_argument("--out_csv", default="/home/student2/jaskra/runs/_small_data_table.csv")
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--require_seeds", default="3", help="how many seeds required per (frac,method), default 3")
    args = ap.parse_args()

    root = Path(args.runs_root)
    req = int(args.require_seeds)

    rows = []
    for run_dir in scan_runs(root, max_depth=args.max_depth):
        name = run_dir.name
        frac = find_frac(name)
        method = find_method(name)
        seed = find_seed(name)
        if frac not in (25, 50, 75):
            continue
        if method is None:
            continue
        if seed is None:
            continue

        metric = None
        # try logs (newest-first)
        logs = candidate_logs(run_dir)
        for lp in logs[:8]:
            metric = extract_metric_from_log(lp)
            if metric is not None:
                break
        if metric is None:
            continue

        rows.append({
            "frac": frac,
            "method": method,
            "seed": seed,
            "run_dir": str(run_dir),
            "metric": metric,
        })

    # group by (frac, method)
    by = {}
    for r in rows:
        by.setdefault((r["frac"], r["method"]), {})[r["seed"]] = r["metric"]

    # keep only groups with enough seeds; pick smallest seed ids deterministically
    agg = []
    missing = []
    for frac in (25, 50, 75):
        for method in ("No-Graph", "Grid-Only", "Dyn-k=16"):
            seeds_map = by.get((frac, method), {})
            if len(seeds_map) < req:
                missing.append((frac, method, sorted(seeds_map.keys())))
                continue
            seeds_sorted = sorted(seeds_map.keys())[:req]
            vals = [seeds_map[s] for s in seeds_sorted]
            mu, sd = mean_std(vals)
            agg.append({
                "frac": frac,
                "method": method,
                "n": req,
                "seeds": ",".join(map(str, seeds_sorted)),
                "mu": mu,
                "sd": sd,
            })

    # compute gains vs no-graph per frac
    mu_no = {a["frac"]: a["mu"] for a in agg if a["method"] == "No-Graph"}
    for a in agg:
        if a["method"] == "No-Graph":
            a["gain"] = None
        else:
            base = mu_no.get(a["frac"], math.nan)
            a["gain"] = (a["mu"] - base) if (not math.isnan(base) and not math.isnan(a["mu"])) else math.nan

    # write CSV
    csv_lines = ["train_frac,method,n,seeds,mean_CpD,std_CpD,gain_vs_no_graph"]
    agg_sorted = sorted(agg, key=lambda x: (x["frac"], ["No-Graph","Grid-Only","Dyn-k=16"].index(x["method"])))
    for a in agg_sorted:
        gain = "" if a["gain"] is None else ("" if math.isnan(a["gain"]) else f"{a['gain']:.4f}")
        csv_lines.append(f"{a['frac']},{a['method']},{a['n']},{a['seeds']},{a['mu']:.6f},{a['sd']:.6f},{gain}")
    Path(args.out_csv).write_text("\n".join(csv_lines), encoding="utf-8")

    # build LaTeX table (your template)
    def row_line(frac, method):
        rec = next((x for x in agg_sorted if x["frac"]==frac and x["method"]==method), None)
        if rec is None:
            val = "XX"
            gain = "--" if method=="No-Graph" else "XX"
            return f"{frac}\\% & {method} & {val} & {gain} \\\\"
        val = fmt_pm(rec["mu"], rec["sd"], nd=4)
        if method == "No-Graph":
            gain = "--"
        else:
            gain = "XX" if (rec["gain"] is None or math.isnan(rec["gain"])) else f"{rec['gain']:+.4f}"
        return f"{frac}\\% & {method} & {val} & {gain} \\\\"

    tex = r"""\begin{table}[t]
\centering
\caption{Small-data robustness on \datasetA{} (val). Mean$\pm$std over 3 seeds.}
\label{tab:small-data}
\begin{tabular}{c l c c}
\toprule
Train frac & Method & Dice(C)$+$Dice(D) $\uparrow$ & Gain vs No-Graph $\uparrow$ \\
\midrule
""" + "\n".join([
        row_line(25, "No-Graph"),
        row_line(25, "Grid-Only"),
        row_line(25, "Dyn-k=16"),
        r"\midrule",
        row_line(50, "No-Graph"),
        row_line(50, "Grid-Only"),
        row_line(50, "Dyn-k=16"),
        r"\midrule",
        row_line(75, "No-Graph"),
        row_line(75, "Grid-Only"),
        row_line(75, "Dyn-k=16"),
    ]) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    Path(args.out_tex).write_text(tex, encoding="utf-8")

    print(f"[OK] wrote LaTeX: {args.out_tex}")
    print(f"[OK] wrote CSV:   {args.out_csv}")
    print(f"[FOUND] rows_used={len(rows)} groups_ok={len(agg_sorted)}")
    if missing:
        print("[MISSING groups] (frac, method, seeds_found):")
        for frac, method, seeds in missing:
            print(f"  {frac}% {method}: seeds={seeds}")

if __name__ == "__main__":
    main()
