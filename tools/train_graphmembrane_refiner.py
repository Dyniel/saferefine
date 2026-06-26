#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from unified_graphseg import GraphSegUnified, Refuge2NPZ, collate, dice_bin, read_idx_file, set_seed, ts  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_all", required=True)
    ap.add_argument("--idx_train", required=True)
    ap.add_argument("--idx_val", required=True)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--dt", type=float, default=0.35)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--base_logit", type=float, default=3.0)
    ap.add_argument("--residual_clip", type=float, default=5.0)
    ap.add_argument("--lambda_energy", type=float, default=0.02)
    ap.add_argument("--lambda_residual", type=float, default=0.01)
    ap.add_argument("--source", type=str, default="synthetic", choices=["synthetic", "host", "mixed"])
    ap.add_argument("--mixed_host_prob", type=float, default=0.5)
    ap.add_argument("--host_ckpt", type=str, default="")
    ap.add_argument("--host_input_mode", type=str, default="hard", choices=["hard", "soft"])
    ap.add_argument("--host_soft_clip", type=float, default=5.0)
    ap.add_argument("--resume_membrane", type=str, default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", type=str, default="runs/graphmembrane")
    ap.add_argument("--run_name", type=str, default="refuge2_graphmembrane")
    ap.add_argument("--amp", type=int, default=0)
    ap.add_argument("--max_train_batches", type=int, default=0)
    ap.add_argument("--max_val_batches", type=int, default=0)
    ap.add_argument("--data_verbose", type=int, default=0)
    return ap.parse_args()


def dice_loss_from_logits(logits, y, include_bg=False):
    probs = F.softmax(logits, dim=1)
    one = F.one_hot(y, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * one).sum(dims)
    den = probs.sum(dims) + one.sum(dims)
    dice = (2.0 * inter + 1e-5) / (den + 1e-5)
    if not include_bg and dice.shape[0] >= 3:
        dice = dice[1:]
    return 1.0 - dice.mean()


def cp_d_from_pred(pred, y):
    p = pred.detach().cpu().numpy()
    g = y.detach().cpu().numpy()
    cup = []
    disc = []
    for i in range(p.shape[0]):
        cup.append(dice_bin(p[i] == 2, g[i] == 2))
        disc.append(dice_bin(p[i] > 0, g[i] > 0))
    return float(np.mean(cup)), float(np.mean(disc)), float(np.mean(cup) + np.mean(disc))


def morph_mask(mask, op, k):
    x = mask.float()
    pad = k // 2
    if op == "dilate":
        return F.max_pool2d(x, kernel_size=k, stride=1, padding=pad) > 0.5
    if op == "erode":
        return (1.0 - F.max_pool2d(1.0 - x, kernel_size=k, stride=1, padding=pad)) > 0.5
    return mask.bool()


def smooth_noise_like(mask, scale=32):
    b, _, h, w = mask.shape
    hh = max(2, h // scale)
    ww = max(2, w // scale)
    z = torch.rand((b, 1, hh, ww), device=mask.device)
    return F.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)


def perturb_labels(y):
    # Produces plausible OD/OC-style nested perturbations: shifts, local dilation/
    # erosion, holes, and spurious boundary bumps. This is the synthetic training
    # distribution for the universal membrane refiner.
    b, h, w = y.shape
    disc = (y > 0).unsqueeze(1)
    cup = (y == 2).unsqueeze(1)

    out_disc = disc.clone()
    out_cup = cup.clone()
    for i in range(b):
        k = random.choice([3, 5, 7])
        op_d = random.choice(["none", "dilate", "erode"])
        op_c = random.choice(["none", "dilate", "erode"])
        out_disc[i:i+1] = morph_mask(out_disc[i:i+1], op_d, k)
        out_cup[i:i+1] = morph_mask(out_cup[i:i+1], op_c, k)

        if random.random() < 0.75:
            dy = random.randint(-8, 8)
            dx = random.randint(-8, 8)
            out_disc[i] = torch.roll(out_disc[i], shifts=(dy, dx), dims=(1, 2))
        if random.random() < 0.75:
            dy = random.randint(-6, 6)
            dx = random.randint(-6, 6)
            out_cup[i] = torch.roll(out_cup[i], shifts=(dy, dx), dims=(1, 2))

    noise_d = smooth_noise_like(out_disc, scale=random.choice([16, 24, 32]))
    noise_c = smooth_noise_like(out_cup, scale=random.choice([16, 24, 32]))
    if random.random() < 0.7:
        out_disc = out_disc ^ ((noise_d > 0.82) & (F.max_pool2d(out_disc.float(), 9, 1, 4) > 0.5))
    if random.random() < 0.7:
        out_cup = out_cup ^ ((noise_c > 0.84) & (F.max_pool2d(out_cup.float(), 7, 1, 3) > 0.5))

    out_cup = out_cup & out_disc
    yc = torch.zeros_like(y)
    yc[out_disc[:, 0] & (~out_cup[:, 0])] = 1
    yc[out_cup[:, 0]] = 2
    return yc


def labels_to_logits(y, num_classes=3, base_logit=3.0):
    one = F.one_hot(y, num_classes=num_classes).permute(0, 3, 1, 2).float()
    return (2.0 * one - 1.0) * float(base_logit)


class MaterialNet(nn.Module):
    def __init__(self, in_ch, hidden=48, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.force = nn.Conv2d(hidden, num_classes, 1)
        self.stiffness = nn.Conv2d(hidden, 1, 1)
        self.damping = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):
        h = self.net(x)
        force = self.force(h)
        stiffness = 0.02 + 0.98 * torch.sigmoid(self.stiffness(h))
        damping = 0.05 + 0.90 * torch.sigmoid(self.damping(h))
        return force, stiffness, damping


class GraphMembraneRefiner(nn.Module):
    def __init__(self, hidden=48, steps=6, dt=0.35, alpha=1.0, residual_clip=5.0):
        super().__init__()
        self.steps = int(steps)
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.residual_clip = float(residual_clip)
        self.material = MaterialNet(in_ch=3 + 3 + 2, hidden=hidden, num_classes=3)
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        self.register_buffer("lap_kernel", lap.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def laplacian(self, q):
        return F.conv2d(q, self.lap_kernel, padding=1, groups=3)

    def forward(self, img, corrupt_logits):
        probs = F.softmax(corrupt_logits, dim=1)
        entropy = -(probs * torch.clamp(probs, min=1e-6).log()).sum(dim=1, keepdim=True) / math.log(3.0)
        boundary = torch.clamp(torch.abs(self.laplacian(probs)).sum(dim=1, keepdim=True), 0.0, 1.0)
        x = torch.cat([img, probs, entropy, boundary], dim=1)
        force, stiffness, damping = self.material(x)

        q = torch.zeros_like(corrupt_logits)
        v = torch.zeros_like(corrupt_logits)
        energy = 0.0
        for _ in range(self.steps):
            smooth_force = stiffness * self.laplacian(q)
            restoring = -0.05 * q
            v = (1.0 - self.dt * damping) * v + self.dt * (force + smooth_force + restoring)
            q = q + self.dt * v
            if self.residual_clip > 0:
                q = torch.clamp(q, -self.residual_clip, self.residual_clip)
            energy = energy + (stiffness * (self.laplacian(q) ** 2)).mean()

        refined = corrupt_logits + self.alpha * q
        aux = {
            "force_abs": force.abs().mean(),
            "stiffness": stiffness.mean(),
            "damping": damping.mean(),
            "energy": energy / max(1, self.steps),
        }
        return refined, q, aux


def make_host_model(args, device):
    if not args.host_ckpt:
        return None
    model = GraphSegUnified(
        backbone="segformer_b0",
        num_classes=3,
        feat_dim=256,
        hidden=512,
        depth=3,
        graph_down=2,
        grid4=True,
        dyn_on="feat",
        dyn_k=16,
        dyn_window=2,
        alpha_graph=0.0,
        graph_output_mode="residual",
    ).to(device)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
        _ = model(dummy, dyn_on_eval="feat", eval_dyn_k=16)
    ckpt = torch.load(args.host_ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{ts()}][host] loaded={args.host_ckpt}", flush=True)
    if missing:
        print(f"[{ts()}][host] missing={len(missing)}", flush=True)
    if unexpected:
        print(f"[{ts()}][host] unexpected={len(unexpected)}", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def host_corruption(host_model, xb, args):
    host_logits, _, _ = host_model(xb, dyn_on_eval="feat", eval_dyn_k=16)
    if args.host_input_mode == "hard":
        yh = torch.argmax(host_logits, dim=1)
        return labels_to_logits(yh, base_logit=args.base_logit), yh
    logits = torch.clamp(host_logits.detach(), -args.host_soft_clip, args.host_soft_clip)
    return logits, torch.argmax(logits, dim=1)


def make_corruption(yb, xb, host_model, args, train):
    use_host = args.source == "host"
    if args.source == "mixed":
        use_host = (random.random() < args.mixed_host_prob) or (not train)
    if use_host:
        if host_model is None:
            raise RuntimeError("--source host/mixed requires --host_ckpt")
        return host_corruption(host_model, xb, args)
    yc = perturb_labels(yb)
    return labels_to_logits(yc, base_logit=args.base_logit), yc


def run_epoch(model, host_model, loader, device, optimizer, scaler, args, train=True):
    model.train(train)
    if host_model is not None:
        host_model.eval()
    sums = defaultdict(float)
    n = 0
    max_batches = args.max_train_batches if train else args.max_val_batches

    for bi, (xb, yb, _meta) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        corrupt_logits, yc = make_corruption(yb, xb, host_model, args, train=train)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(args.amp) and device.type == "cuda"):
            refined, _q, aux = model(xb, corrupt_logits)
            ce = F.cross_entropy(refined, yb)
            dl = dice_loss_from_logits(refined, yb)
            residual = torch.clamp(refined - corrupt_logits, -args.residual_clip, args.residual_clip)
            loss = ce + dl + args.lambda_energy * aux["energy"] + args.lambda_residual * residual.abs().mean()

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            pred_in = torch.argmax(corrupt_logits, dim=1)
            pred_out = torch.argmax(refined, dim=1)
            in_c, in_d, in_cpd = cp_d_from_pred(pred_in, yb)
            out_c, out_d, out_cpd = cp_d_from_pred(pred_out, yb)

        bs = xb.size(0)
        n += bs
        sums["loss"] += float(loss.detach().cpu()) * bs
        sums["in_cpd"] += in_cpd * bs
        sums["out_cpd"] += out_cpd * bs
        sums["out_c"] += out_c * bs
        sums["out_d"] += out_d * bs
        sums["gain"] += (out_cpd - in_cpd) * bs
        sums["force_abs"] += float(aux["force_abs"].detach().cpu()) * bs
        sums["stiffness"] += float(aux["stiffness"].detach().cpu()) * bs
        sums["damping"] += float(aux["damping"].detach().cpu()) * bs

        if max_batches > 0 and (bi + 1) >= max_batches:
            break

    return {k: v / max(1, n) for k, v in sums.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_dir) / args.run_name
    (out_root / "ckpt").mkdir(parents=True, exist_ok=True)

    idx_tr = read_idx_file(args.idx_train)
    idx_va = read_idx_file(args.idx_val)
    train_ds = Refuge2NPZ(args.npz_all, idx_tr, img_size=args.img_size, train=True, verbose=bool(args.data_verbose))
    val_ds = Refuge2NPZ(args.npz_all, idx_va, img_size=args.img_size, train=False, verbose=bool(args.data_verbose))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False, collate_fn=collate)

    model = GraphMembraneRefiner(
        hidden=args.hidden,
        steps=args.steps,
        dt=args.dt,
        alpha=args.alpha,
        residual_clip=args.residual_clip,
    ).to(device)
    if args.resume_membrane:
        ckpt = torch.load(args.resume_membrane, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"[{ts()}][membrane] resumed={args.resume_membrane}", flush=True)

    host_model = make_host_model(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")

    print(f"[{ts()}][cfg] device={device} img_size={args.img_size} batch={args.batch}", flush=True)
    print(f"[{ts()}][cfg] hidden={args.hidden} steps={args.steps} dt={args.dt} alpha={args.alpha} clip={args.residual_clip}", flush=True)
    print(f"[{ts()}][cfg] source={args.source} host_input_mode={args.host_input_mode} lambda_energy={args.lambda_energy} lambda_residual={args.lambda_residual}", flush=True)
    print(f"[{ts()}][cfg] out={out_root}", flush=True)

    best = -1e9
    history = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, host_model, train_loader, device, opt, scaler, args, train=True)
        va = run_epoch(model, host_model, val_loader, device, None, scaler, args, train=False)
        sched.step()
        score = va["out_cpd"]
        row = {"epoch": ep, "train": tr, "val": va, "lr": opt.param_groups[0]["lr"]}
        history.append(row)

        print(
            f"{ts()} Ep {ep:03d} | "
            f"TR loss={tr['loss']:.4f} gain={tr['gain']:+.4f} | "
            f"VA in={va['in_cpd']:.4f} out={va['out_cpd']:.4f} gain={va['gain']:+.4f} "
            f"C={va['out_c']:.4f} D={va['out_d']:.4f} | "
            f"k={va['stiffness']:.3f} damp={va['damping']:.3f} | {(time.time()-t0)/60:.2f} min",
            flush=True,
        )

        if score > best:
            best = score
            ck = {
                "epoch": ep,
                "state_dict": model.state_dict(),
                "best_score": best,
                "args": vars(args),
                "history": history,
            }
            torch.save(ck, out_root / "ckpt" / "best.pt")
            print(f"{ts()} [CKPT] new best out_CpD={best:.6f} gain={va['gain']:+.6f}", flush=True)

    (out_root / "history.json").write_text(json.dumps(history, indent=2))
    print(f"{ts()} DONE best_out_CpD={best:.6f} ckpt={out_root / 'ckpt' / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
