#!/usr/bin/env python3
"""保存済みsubset統計を、episodeごとの平均・分散を結合する別計算で検証する。"""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
SOURCE = Path.home() / "projects/demo-d3"

def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def main():
    manifest = json.loads((ROOT / "subsets/manifest.json").read_text())
    cfg = json.loads((ROOT / "experiment_conditions.p0_input.json").read_text())
    episode_data = {}
    for shard in manifest["source_shards"]:
        p = Path(shard["path"]).expanduser()
        assert digest(p) == shard["sha256"]
        # 画像は準備時に全件デコード済み。再検証では必要な数値列だけ読む。
        table = pq.read_table(p, columns=["episode_index", "observation.state", "action"])
        ids = np.array(table["episode_index"])
        for episode in shard["selected_episode_ids"]:
            rows = np.flatnonzero(ids == episode)
            episode_data[episode] = {}
            for key in ["observation.state", "action"]:
                values = np.array(table[key].take(rows).to_pylist(), dtype=np.float64)
                episode_data[episode][key] = (len(values), values.mean(0), values.var(0, ddof=0))
    maximum_error = 0.0
    checked = []
    for item in manifest["subsets"]:
        rep = next(r for r in cfg["replicates"] if r["id"] == item["replicate"])
        expected_ids = rep["N20" if item["condition"] == "N20B" else item["condition"]]
        assert item["episode_ids"] == expected_ids
        stats_path = ROOT / item["stats_path"]
        selection_path = ROOT / item["selection_path"]
        assert digest(stats_path) == item["stats_sha256"]
        assert digest(selection_path) == item["selection_sha256"]
        stats = json.loads(stats_path.read_text())
        selection = json.loads(selection_path.read_text())
        assert selection["episode_ids"] == expected_ids
        for key in ["observation.state", "action"]:
            # 平均・二乗偏差和をepisode単位で結合。単純なepisode平均ではない。
            count, mean, m2 = 0, np.zeros(len(stats[key]["mean"])), np.zeros(len(stats[key]["mean"]))
            for episode in expected_ids:
                n, ep_mean, ep_var = episode_data[episode][key]
                delta = ep_mean - mean
                combined = count + n
                m2 += n * ep_var + delta**2 * count * n / combined
                mean += delta * n / combined
                count = combined
            std = np.sqrt(m2 / count)
            assert count == item["frame_count"] == selection["frame_count"] == stats[key]["count"][0]
            for calculated, saved in [(mean, stats[key]["mean"]), (std, stats[key]["std"])]:
                error = float(np.max(np.abs(calculated - saved)))
                maximum_error = max(maximum_error, error)
                assert np.allclose(calculated, saved, rtol=0, atol=1e-12)
        checked.append(f'{item["replicate"]}/{item["condition"]}')
    for item in json.loads((ROOT / "evidence/source_evidence_hashes.json").read_text())["files"]:
        assert digest(Path(item["path"]).expanduser()) == item["sha256"]
    inventory = json.loads((ROOT / "evidence/source_inventory.json").read_text())
    for item in inventory["base_weights"]:
        assert digest(Path(item["path"]).expanduser()) == item["sha256"]
    git = ["git", "--no-optional-locks", "-C", str(SOURCE / "lerobot")]
    assert subprocess.check_output(git + ["status", "--porcelain=v1", "--untracked-files=all"], text=True) == ""
    assert subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip() == cfg["source"]["lerobot_commit"]
    result = {"status": "PASS", "checked_subsets": checked, "mean_std_max_abs_error": maximum_error,
              "statistics_atol": 1e-12, "original_32_shards_and_base_weights_unchanged": True,
              "reviewed_source_and_old_log_hashes_unchanged": True, "git_status_clean": True,
              "method": "Parallel population-variance merge across actual episode rows; independent of concatenation-based preparation."}
    (ROOT / "evidence/final_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
