# Adapter for tools.infer_any_dataset: it expects `GraphSeg` symbol in model_py.
# We map it to the training-time model `HRNetGraphSeg` (defined in hrnet_graphseg_min.py).

from __future__ import annotations
from pathlib import Path
import torch

# Your training-time implementation (should contain HRNetGraphSeg)
from tools.hrnet_graphseg_min import HRNetGraphSeg

class GraphSeg(HRNetGraphSeg):
    """
    Compatibility alias: tools.infer_any_dataset looks for GraphSeg.
    Inherits the real model implementation.
    """
    pass

def build_from_ckpt(ckpt_path: str, device: str = "cpu", **override_kwargs):
    """
    Optional helper: build model using args stored in checkpoint (if present).
    `tools.infer_any_dataset` might not call it, but it's useful and harmless.
    """
    ckpt_path = str(ckpt_path)
    ck = torch.load(ckpt_path, map_location="cpu")

    # default hyperparams (match your training defaults)
    kwargs = dict(
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
    )

    a = ck.get("args", None)
    if isinstance(a, dict):
        # pull known fields if present
        for k in list(kwargs.keys()):
            if k in a:
                kwargs[k] = a[k]
        # some checkpoints store these under slightly different names
        if "alpha_graph" in a:
            kwargs["alpha_graph"] = a["alpha_graph"]
        if "dyn_on" in a:
            kwargs["dyn_on"] = a["dyn_on"]
        if "dyn_k" in a:
            kwargs["dyn_k"] = int(a["dyn_k"])
        if "dyn_window" in a:
            kwargs["dyn_window"] = int(a["dyn_window"])
        if "grid4" in a:
            kwargs["grid4"] = bool(int(a["grid4"])) if isinstance(a["grid4"], (int, str)) else bool(a["grid4"])
        if "graph_down" in a:
            kwargs["graph_down"] = int(a["graph_down"])
        if "backbone" in a:
            kwargs["backbone"] = a["backbone"]
        if "feat_dim" in a:
            kwargs["feat_dim"] = int(a["feat_dim"])
        if "hidden" in a:
            kwargs["hidden"] = int(a["hidden"])
        if "depth" in a:
            kwargs["depth"] = int(a["depth"])

    # apply overrides from caller
    kwargs.update(override_kwargs)

    m = GraphSeg(**kwargs)
    sd = ck.get("state_dict", ck)
    m.load_state_dict(sd, strict=True)
    m.to(device)
    m.eval()
    return m

# alias some likely names (different helpers might look for these)
build_model = build_from_ckpt
build_model_from_ckpt = build_from_ckpt
