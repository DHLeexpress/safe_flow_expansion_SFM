import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_promoted_checkpoints_are_exact():
    assert sha256(ROOT / "checkpoints/hp10_pretrained_r0.pt") == (
        "1b5179c935d3eeff8824967d707d64cc9bab273949ee1f0e4f190172bab1b215"
    )
    assert sha256(ROOT / "checkpoints/b1_legacy_selected_A_r10.pt") == (
        "bf6f521dd2dd6de4cffcce672a8ce4adbf00bb14e71dd9fd27704d205f65744c"
    )


def test_completed_banks_are_matched_and_severe_ood():
    matched = json.loads((ROOT / "provenance/pre_expansion/matched_id_metrics.json").read_text())
    shifted = json.loads((ROOT / "provenance/pre_expansion/double_shift_ood_metrics.json").read_text())
    assert matched["environment"]["scene_profile"] == "matched_id"
    assert matched["environment"]["n_ped"] == 20
    assert shifted["environment"]["scene_profile"] == "double_density_velocity_ood"
    assert shifted["environment"]["n_ped"] == 40
    assert shifted["environment"]["ped_speed_range"] == [1.0, 2.0]
    assert matched["M_per_gamma"] == shifted["M_per_gamma"] == 100


def test_current_result_is_not_claimed_complete():
    recipe = json.loads((ROOT / "configs/current_sweep_recipe.json").read_text())
    assert recipe["status"] == "RUNNING_NOT_A_RESULT"
    gate = json.loads((ROOT / "provenance/runtime_gate/RUNTIME_FORECAST.json").read_text())
    assert gate["status"] == "RUNTIME_GATE_PASS"
    assert gate["source_commit"] == recipe["source_commit"]


def test_manifest_covers_documentation_and_source():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    paths = {row["path"] for row in manifest["files"]}
    assert "README.md" in paths
    assert "source_snapshot/overnight_run_07_12_sfm/sfm_b1_expand.py" in paths
    assert "assets/results/pre_expansion/double_shift_ood_metrics.png" in paths

