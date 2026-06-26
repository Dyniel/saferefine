# minimal HRNetGraphSeg for evaluation (no training)
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception as e:
    raise RuntimeError("Brak timm (pip install timm).") from e

try:
    from torch_geometric.nn import SAGEConv
except Exception as e:
    raise RuntimeError("Brak torch_geometric (PyG).") from e


def build_grid_edges(h, w, grid4=True, device="cpu"):
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if not grid4:
        dirs = dirs + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    edges = []
    def nid(r, c): return r * w + c
    for r in range(h):
        for c in range(w):
            i = nid(r, c)
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    edges.append([i, nid(rr, cc)])
    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    e = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    rev = e[[1, 0], :]
    return torch.cat([e, rev], dim=1)


def build_window_candidates(h, w, r, device):
    N = h * w
    offs = []
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if dr == 0 and dc == 0:
                continue
            offs.append((dr, dc))

    cand = []
    for rr in range(h):
        for cc in range(w):
            neigh = []
            for dr, dc in offs:
                r2, c2 = rr + dr, cc + dc
                if 0 <= r2 < h and 0 <= c2 < w:
                    neigh.append(r2 * w + c2)
            if len(neigh) == 0:
                neigh = [rr * w + cc]
            cand.append(neigh)

    M = max(len(x) for x in cand)
    out = torch.empty((N, M), dtype=torch.long, device=device)
    for i in range(N):
        row = cand[i]
        if len(row) < M:
            row = row + [row[-1]] * (M - len(row))
        out[i] = torch.tensor(row, dtype=torch.long, device=device)
    return out


def build_dyn_edges_local_knn(feat, cand_idx, k):
    N, D = feat.shape
    M = cand_idx.shape[1]
    f = F.normalize(feat, dim=1)
    cand = f[cand_idx]
    sim = (f[:, None, :] * cand).sum(dim=-1)
    k_eff = min(int(k), int(M))
    top = torch.topk(sim, k_eff, dim=1).indices
    nbr = cand_idx.gather(1, top)
    src = torch.arange(N, device=feat.device, dtype=torch.long)[:, None].expand(N, k_eff).reshape(-1)
    dst = nbr.reshape(-1)
    e = torch.stack([src, dst], dim=0)
    rev = e[[1, 0], :]
    return torch.cat([e, rev], dim=1)


class GraphRefiner(nn.Module):
    def __init__(self, in_dim, hidden=512, depth=3, out_dim=3, dropout=0.1):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.drop = nn.Dropout(dropout)
        self.lin_out = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index):
        h = self.lin_in(x)
        for conv, norm in zip(self.convs, self.norms):
            h2 = conv(h, edge_index)
            h2 = F.gelu(h2)
            h2 = norm(h2)
            h = h + self.drop(h2)
        return self.lin_out(h), h


def make_backbone(name: str):
    # timm HRNet: "hrnet_w48" etc.
    m = timm.create_model(name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
    return m


class HRNetGraphSeg(nn.Module):
    def __init__(
        self,
        backbone="hrnet_w48",
        num_classes=3,
        feat_dim=256,
        hidden=512,
        depth=3,
        graph_down=2,
        grid4=True,
        dyn_on="feat",
        dyn_k=4,
        dyn_window=2,
        alpha_graph=0.55,
        run_name="",
        out_dir="",
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = make_backbone(backbone)
        chs = self.backbone.feature_info.channels()
        self.high_idx = int(np.argmin(self.backbone.feature_info.reduction()))
        high_ch = chs[self.high_idx]

        self.feat_proj = nn.Sequential(
            nn.Conv2d(high_ch, feat_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.GELU(),
        )
        self.seg_head = nn.Conv2d(feat_dim, num_classes, 1)

        self.graph_down = int(graph_down)
        self.grid4 = bool(grid4)
        self.dyn_on = str(dyn_on)
        self.dyn_k = int(dyn_k)
        self.dyn_window = int(dyn_window)
        self.alpha_graph = float(alpha_graph)

        self.graph = GraphRefiner(in_dim=feat_dim, hidden=hidden, depth=depth, out_dim=num_classes, dropout=0.1)

        rn = (str(run_name) + " " + str(out_dir)).strip()
        self._graph_enabled = not bool(re.search(r"(^|[ _/-])no_graph($|[ _/-])", rn))

        self._cache = {}

    def _cached(self, key, fn):
        v = self._cache.get(key, None)
        if v is None:
            v = fn()
            self._cache[key] = v
        return v

    def forward(self, x, dyn_on_eval=None, eval_dyn_k=None):
        B, _, H, W = x.shape
        feats = self.backbone(x)
        f = feats[self.high_idx]
        f = self.feat_proj(f)
        logits = self.seg_head(f)
        logits_up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        if (self.alpha_graph <= 0) or (not self._graph_enabled):
            return logits_up, logits_up, None

        if self.graph_down > 1:
            f_g = F.avg_pool2d(f, kernel_size=self.graph_down, stride=self.graph_down)
            logits_g = F.avg_pool2d(logits, kernel_size=self.graph_down, stride=self.graph_down)
        else:
            f_g = f
            logits_g = logits

        _, Cg, Hg, Wg = f_g.shape
        N = Hg * Wg

        f_nodes = f_g.flatten(2).permute(0, 2, 1).contiguous()
        f_nodes_all = f_nodes.view(B * N, Cg)

        edge_base = self._cached(
            ("grid", Hg, Wg, self.grid4, x.device.type),
            lambda: build_grid_edges(Hg, Wg, grid4=self.grid4, device=x.device)
        )

        if dyn_on_eval is None:
            dyn_on_eval = self.dyn_on
        if eval_dyn_k is None:
            eval_dyn_k = self.dyn_k

        use_dyn = (dyn_on_eval in ("feat", "feature", "features")) and (int(eval_dyn_k) > 0)

        if use_dyn:
            cand = self._cached(
                ("cand", Hg, Wg, self.dyn_window, x.device.type),
                lambda: build_window_candidates(Hg, Wg, self.dyn_window, device=x.device)
            )

        edges = []
        for b in range(B):
            e = edge_base + b * N
            if use_dyn:
                dyn = build_dyn_edges_local_knn(f_nodes[b], cand, k=int(eval_dyn_k)) + b * N
                e = torch.cat([e, dyn], dim=1)
            edges.append(e)

        edge_index = torch.cat(edges, dim=1) if len(edges) else torch.zeros((2, 0), dtype=torch.long, device=x.device)

        node_logits_all, _ = self.graph(f_nodes_all, edge_index)
        node_logits = node_logits_all.view(B, N, -1).permute(0, 2, 1).contiguous().view(B, -1, Hg, Wg)
        node_logits_up = F.interpolate(node_logits, size=(H, W), mode="bilinear", align_corners=False)

        out = (1.0 - self.alpha_graph) * logits_up + self.alpha_graph * node_logits_up
        return out, logits_up, node_logits_up


def build_hrnet_graphseg_from_ckpt(ckpt, device="cuda"):
    # --- PATCH: accept ckpt path or loaded dict ---
    from pathlib import Path as _Path
    import torch as _torch
    if isinstance(ckpt, (str, _Path)):
        ckpt = _torch.load(str(ckpt), map_location="cpu")
    # ------------------------------------------------
    args = ckpt.get("args", {}) or {}
    bb = str(args.get("backbone", "hrnet_w48"))
    model = HRNetGraphSeg(
        backbone=bb,
        num_classes=3,
        feat_dim=int(args.get("feat_dim", 256)),
        hidden=int(args.get("hidden", 512)),
        depth=int(args.get("depth", 3)),
        graph_down=int(args.get("graph_down", 2)),
        grid4=bool(int(args.get("grid4", 1))),
        dyn_on=str(args.get("dyn_on", "feat")),
        dyn_k=int(args.get("dyn_k", 4)),
        dyn_window=int(args.get("dyn_window", 2)),
        alpha_graph=float(args.get("alpha_graph", 0.55)),
        run_name=str(args.get("run_name","")),
        out_dir=str(args.get("out_dir","")),
    ).to(device)

    sd = ckpt.get("state_dict", None)
    if sd is None:
        raise RuntimeError("ckpt nie ma state_dict")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model, missing, unexpected

if __name__ == "__main__":
    main()
