import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_synthetic_portfolio_csv(path: Path) -> None:
    fieldnames = [
        "id",
        "idx",
        "method",
        "alpha",
        "cup",
        "disc",
        "cpd",
        "changed",
        "force_abs",
        "stiffness",
        "damping",
        "disc_area_delta",
        "cup_area_delta",
        "cup_disc_ratio_delta",
        "disc_centroid_shift",
        "cup_centroid_shift",
        "component_delta",
        "disc_changed",
        "cup_changed",
        "geom_risk",
        "host_entropy",
        "host_confidence",
        "host_margin",
        "quality_risk",
    ]
    rows = []
    # Eight images are enough to exercise calibration/test splitting and the CRC
    # host-fallback path. The candidate action is intentionally harmful on the
    # calibration half, so it should not satisfy the conservative UCB contract.
    for i in range(8):
        image_id = f"case_{i:03d}"
        host_cpd = 0.80
        bad_cpd = 0.70 if i < 4 else 0.85
        base = {
            "id": image_id,
            "idx": i,
            "alpha": 1.0,
            "cup": 0.0,
            "disc": 0.0,
            "force_abs": 0.0,
            "stiffness": 0.0,
            "damping": 0.0,
            "disc_area_delta": 0.0,
            "cup_area_delta": 0.0,
            "cup_disc_ratio_delta": 0.0,
            "disc_centroid_shift": 0.0,
            "cup_centroid_shift": 0.0,
            "component_delta": 0.0,
            "disc_changed": 0.0,
            "cup_changed": 0.0,
            "geom_risk": 0.0,
            "host_entropy": 0.0,
            "host_confidence": 1.0,
            "host_margin": 1.0,
            "quality_risk": 0.0,
        }
        rows.append({**base, "method": "host", "cpd": host_cpd, "changed": 0.0})
        rows.append({**base, "method": "host_morph", "cpd": bad_cpd, "changed": 0.1})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_crc_falls_back_to_host_on_uncertified_action(tmp_path: Path) -> None:
    csv_path = tmp_path / "portfolio.csv"
    out_json = tmp_path / "summary.json"
    _write_synthetic_portfolio_csv(csv_path)

    repo = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "eval_safe_action_portfolio.py"),
            "--input_csv",
            f"synthetic={csv_path}",
            "--risk_score",
            "changed",
            "--tail_constraint_mode",
            "full",
            "--bound_mode",
            "hoeffding",
            "--out_summary",
            str(out_json),
        ],
        check=True,
        cwd=repo,
    )

    payload = json.loads(out_json.read_text())
    assert payload["crc_best"]["selected_action"] == "host"
    assert payload["crc_best"]["mean_gain"] == 0.0
    assert payload["crc_best"]["mean_harm"] == 0.0
