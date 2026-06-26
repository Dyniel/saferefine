"""Smoke test of the SafeRefine primary claim (CPU-only, deterministic).

Runs the calibration controller on each committed per-image action CSV under the
paper's conservative tail-risk contract and asserts that the certified policy
(``crc_portfolio``) selects the host action with exactly zero gain and zero harm.
This is the paper's primary result: no tested refiner is certifiable on the
available calibration evidence, so the controller returns the host.

Run with:  python -m pytest tests/ -q
"""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO / "results" / "refiner_zoo_uncert"

# (label, split_group) for the eight paper settings.
SETTINGS = [
    ("isic2018_task1_mediafinal_unet_e120_zoo", "image"),
    ("kvasir_seg_mediafinal_graphseg_e120_zoo", "image"),
    ("kvasir_seg_mediafinal_unet_e120_zoo", "image"),
    ("ph2_mediafinal_unet_e120_zoo", "image"),
    ("polyps_official_mediafinal_graphseg_e120_zoo", "image"),
    ("polyps_official_mediafinal_unet_e120_zoo", "image"),
    ("msd_heart_mri_mediafinal_graphseg_mri_e120_zoo", "patient"),
    ("msd_heart_mri_mediafinal_unet_mri_e120_zoo", "patient"),
]


def run_controller(label, split_group, tmp_path):
    csv_in = INPUT_DIR / f"{label}.csv"
    assert csv_in.exists(), f"missing committed input CSV: {csv_in}"
    out_csv = tmp_path / f"{label}.csv"
    cmd = [
        sys.executable, str(REPO / "tools" / "eval_safe_action_portfolio.py"),
        "--inputs", f"{label}={csv_in}",
        "--risk_score", "host_uncertainty",
        "--split_group", split_group,
        "--selection_mode", "joint",
        "--harm_eps", "0.0",
        "--max_cal_harm_rate", "0.25",
        "--max_cal_drop05_rate", "0.10",
        "--max_cal_mean_harm", "0.02",
        "--bound_mode", "hoeffding",
        "--tail_constraint_mode", "full",
        "--out_csv", str(out_csv),
        "--out_summary", str(tmp_path / f"{label}.json"),
    ]
    subprocess.run(cmd, check=True, cwd=REPO)
    with out_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    crc = [r for r in rows if r["split"] == "test" and r["policy"] == "crc_portfolio"]
    assert len(crc) == 1, f"expected one crc_portfolio test row, got {len(crc)}"
    return crc[0]


@pytest.mark.parametrize("label,split_group", SETTINGS, ids=[s[0] for s in SETTINGS])
def test_primary_contract_selects_host(label, split_group, tmp_path):
    row = run_controller(label, split_group, tmp_path)
    assert row["selected_action"] == "host", (
        f"{label}: expected host fallback under the conservative contract, "
        f"got {row['selected_action']}"
    )
    assert abs(float(row["mean_gain"])) < 1e-9, f"{label}: host gain must be 0"
    assert abs(float(row["mean_harm"])) < 1e-9, f"{label}: host harm must be 0"
    assert abs(float(row["reverted_rate"]) - 1.0) < 1e-9, f"{label}: revert-rate must be 1.0"
