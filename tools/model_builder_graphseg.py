import importlib
import torch.nn as nn

def build_model_from_args(args_dict):
    mod_name = "train_refuge2_npz_unified"
    m = importlib.import_module(mod_name)
    GraphSeg = getattr(m, "GraphSeg")

    model = GraphSeg(
        backbone=args_dict.get("backbone", "segformer_b0"),
        num_classes=3,
        feat_dim=int(args_dict.get("feat_dim", 256)),
        hidden=int(args_dict.get("hidden", 512)),
        depth=int(args_dict.get("depth", 3)),
        graph_down=int(args_dict.get("graph_down", 2)),
        grid4=bool(args_dict.get("grid4", 1)),
        dyn_on=str(args_dict.get("dyn_on", "feat")),
        dyn_k=int(args_dict.get("dyn_k", 16)),
        dyn_window=int(args_dict.get("dyn_window", 2)),
        alpha_graph=float(args_dict.get("alpha_graph", 0.0)),
    )

    if not isinstance(model, nn.Module):
        raise TypeError("build_model_from_args() did not return nn.Module")
    return model
