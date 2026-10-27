#!/usr/bin/env python3
"""P0専用。原本を読み取り、episode選択表と統計だけをこのフォルダへ保存する。"""
from pathlib import Path
import hashlib
import io
import json
import os
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = Path.home() / "projects/demo-d3"
DATA = SOURCE / "lerobot_cache/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e"
MODEL = SOURCE / "hf_cache/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
TRAIN = SOURCE / "outputs/phase2_lora_r16_s30000_46ep/checkpoints/030000/pretrained_model/train_config.json"
OLD_MANIFEST = SOURCE / "repro/phase2_selected_shards.sha256"
EXPECTED_COMMIT = "7427f31801f944f5ec9ce53cda02862ce7d1638b"
CAMERAS = ["observation.images.image", "observation.images.image2"]
TASK = "pick up the black bowl from table center and place it on the plate"

def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def short(path):
    return str(path).replace(str(Path.home()), "~")

def save_json(path, value):
    # 保存先はこの新規作業フォルダ内だけ。キャッシュや旧checkpointには書かない。
    assert path.resolve().is_relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def require(ok, reason):
    if not ok:
        raise RuntimeError("STOP: " + reason)

def main():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU専用環境変数が必要")
    require(ROOT == Path.home() / "projects/pai-final-smolvla", "保存先が指定と異なる")
    require(shutil.disk_usage(ROOT).free >= 50 * 2**30, "空き容量が50 GiB未満")
    cmd = ["git", "--no-optional-locks", "-C", str(SOURCE / "lerobot")]
    require(subprocess.check_output(cmd + ["rev-parse", "HEAD"], text=True).strip() == EXPECTED_COMMIT, "HEAD不一致")
    require(subprocess.check_output(cmd + ["status", "--porcelain=v1", "--untracked-files=all"], text=True) == "", "作業ツリー非clean")
    cfg = json.loads((ROOT / "experiment_conditions.p0_input.json").read_text())
    train = json.loads(TRAIN.read_text())
    source_ids = cfg["source"]["episode_ids"]
    require(source_ids == train["dataset"]["episodes"], "46 episode IDがtrain_configと不一致")
    require(len(source_ids) == len(set(source_ids)) == 46, "46 episodeの件数不正")
    require(MODEL.is_dir() and DATA.is_dir(), "固定snapshot欠落")
    require(sha256(OLD_MANIFEST) == "60682c35ef049425940f5f4906bbab1712ed605d0bcf11590f0617e62c80b240", "旧manifestのhash不一致")
    for rep in cfg["replicates"]:
        sets = [set(rep[name]) for name in ["N10", "N20", "N46"]]
        require([len(x) for x in sets] == [10, 20, 46], "subset件数不正")
        require(sets[0] < sets[1] < sets[2] == set(source_ids), "subset包含関係不正")
        require(all(len(rep[k]) == len(set(rep[k])) for k in ["N10", "N20", "N46"]), "重複episode")
    manifest_rows = [line.split(maxsplit=1) for line in OLD_MANIFEST.read_text().splitlines() if line.strip()]
    require(len(manifest_rows) == 32, "旧manifestのshard数が32ではない")
    # すべての原本が存在しhashが合うことを、subsetを出力する前に確かめる。
    shard_info = []
    for expected_hash, original in manifest_rows:
        path = Path(original)
        require(path.is_relative_to(DATA), "manifestが固定snapshot外を参照")
        require(path.is_file(), f"shard欠落: {short(path)}")
        actual_hash = sha256(path)
        require(actual_hash == expected_hash, f"shard hash不一致: {short(path)}")
        shard_info.append({"path": short(path), "sha256": actual_hash, "bytes": path.stat().st_size})
    tasks = pq.read_table(DATA / "meta/tasks.parquet").to_pandas()
    task_rows = tasks[tasks.index == TASK]
    if len(task_rows) != 1:
        # parquetによってタスク文はindexではなく通常の列に格納される。
        task_rows = tasks[[any(str(v) == TASK for v in row) for row in tasks.to_numpy()]]
    require(len(task_rows) == 1, "対象タスクの辞書照合に失敗")
    target_task_index = int(task_rows["task_index"].iloc[0])
    selected = {}
    for shard, (_, original) in zip(shard_info, manifest_rows, strict=True):
        path = Path(original)
        # metadataのfile_indexは使用せず、実データのepisode_indexで行を選択する。
        table = pq.read_table(path)
        ids = np.asarray(table["episode_index"])
        shard["all_episode_ids"] = sorted(set(ids.tolist()))
        chosen = sorted(set(ids.tolist()) & set(source_ids))
        shard["selected_episode_ids"] = chosen
        for episode in chosen:
            require(episode not in selected, f"episode {episode}が複数shardに出現")
            row_indices = np.flatnonzero(ids == episode)
            rows = table.take(pa.array(row_indices))
            state = np.asarray(rows["observation.state"].to_pylist(), dtype=np.float64)
            action = np.asarray(rows["action"].to_pylist(), dtype=np.float64)
            n = rows.num_rows
            require(state.shape == (n, 8) and action.shape == (n, 7), "state/action次元不正")
            require(np.isfinite(state).all() and np.isfinite(action).all(), "非有限数")
            require(set(rows["task_index"].to_pylist()) == {target_task_index}, "タスク不一致")
            frame_indices = np.asarray(rows["frame_index"])
            require(np.array_equal(frame_indices, np.arange(n)), "フレームの欠落・重複・順序不正")
            # 全選択フレームの2画像をCPUでデコードし、形状と画像データの実在を検査する。
            for camera in CAMERAS:
                require(camera in rows.column_names, "カメラ列欠落")
                for item in rows[camera].to_pylist():
                    require(item is not None and bool(item["bytes"]), "埋め込み画像欠落")
                    with Image.open(io.BytesIO(item["bytes"])) as image:
                        image.load()
                        require(image.size == (256, 256) and image.mode == "RGB", "画像形式不正")
            selected[episode] = {"state": state, "action": action, "frames": n,
                                 "shard": shard["path"], "row_indices": row_indices.tolist()}
        print(f"checked {path.name}: {len(chosen)} selected episodes", flush=True)
    require(set(selected) == set(source_ids), "実ファイルで46 episodeが揃わない")
    require(sum(x["frames"] for x in selected.values()) == 5343, "全教示frame数が既存記録と不一致")
    global_stats = json.loads((DATA / "meta/stats.json").read_text())
    base_sha = sha256(MODEL / "model.safetensors")
    result = {"schema_version": 1, "phase": "P0_CPU_ONLY", "source_episode_ids": source_ids,
              "source_frame_count": 5343, "dataset_revision": DATA.name,
              "base_model_revision": MODEL.name, "base_model_sha256": base_sha,
              "input_json_sha256": sha256(ROOT / "experiment_conditions.p0_input.json"),
              "train_config_sha256": sha256(TRAIN), "old_shard_manifest_sha256": sha256(OLD_MANIFEST),
              "task": TASK, "task_index": target_task_index, "camera_keys": CAMERAS,
              "image_shape_hwc": [256, 256, 3], "all_selected_images_decoded": 5343 * 2,
              "state_dim": 8, "action_dim": 7, "variance_ddof": 0,
              "statistics_accumulation_dtype": "float64", "visual_normalization": "IDENTITY",
              "source_shards": shard_info, "subsets": []}
    n46_stats = None
    for rep in cfg["replicates"]:
        hashes = {}
        for condition in ["N10", "N20", "N46", "N20B"]:
            ids = rep["N20" if condition == "N20B" else condition]
            episode_map = [{"episode_index": e, "frame_count": selected[e]["frames"],
                            "source_shard": selected[e]["shard"], "source_row_indices": selected[e]["row_indices"]}
                           for e in ids]
            frame_count = sum(selected[e]["frames"] for e in ids)
            stats = {}
            for feature, key in [("observation.state", "state"), ("action", "action")]:
                values = np.concatenate([selected[e][key] for e in ids])
                # フレームを等しい重みで扱う母標準偏差。episode平均の平均は使わない。
                stats[feature] = {"mean": values.mean(axis=0).tolist(),
                                  "std": values.std(axis=0, ddof=0).tolist(),
                                  "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
                                  "count": [frame_count]}
            destination = ROOT / "subsets" / rep["id"] / condition
            save_json(destination / "stats.json", stats)
            save_json(destination / "selection.json", {"episode_ids": ids, "frame_count": frame_count,
                      "task": TASK, "task_index": target_task_index, "episodes": episode_map})
            hashes[condition] = (sha256(destination / "stats.json"), sha256(destination / "selection.json"))
            result["subsets"].append({"replicate": rep["id"], "condition": condition, "episode_ids": ids,
                "episode_count": len(ids), "frame_count": frame_count,
                "stats_path": str((destination / "stats.json").relative_to(ROOT)),
                "stats_sha256": hashes[condition][0],
                "selection_path": str((destination / "selection.json").relative_to(ROOT)),
                "selection_sha256": hashes[condition][1], "base_model_sha256": base_sha,
                "same_subset_and_stats_as": "N20" if condition == "N20B" else None})
            if condition == "N46":
                require(n46_stats is None or n46_stats == stats, "RごとのN46統計不一致")
                n46_stats = stats
        require(hashes["N20"] == hashes["N20B"], "N20/N20Bのsubset/stat hash不一致")
    comparison = {"matches_distributed_stats": True, "atol": 1e-6, "rtol": 0,
                  "subset_frames": 5343, "distributed_frames": global_stats["action"]["count"][0],
                  "reason": "N46 is one task's 46 episodes/5343 frames; distributed statistics cover 273465 frames across the dataset.",
                  "features": {}}
    for feature in ["observation.state", "action"]:
        comparison["features"][feature] = {}
        for key in ["mean", "std"]:
            actual = np.array(n46_stats[feature][key])
            supplied = np.array(global_stats[feature][key])
            equal = bool(np.allclose(actual, supplied, atol=1e-6, rtol=0))
            comparison["matches_distributed_stats"] &= equal
            comparison["features"][feature][key] = {"recomputed": actual.tolist(), "distributed": supplied.tolist(),
                     "difference": (actual - supplied).tolist(), "max_abs_difference": float(np.max(np.abs(actual - supplied))),
                     "matches": equal}
    save_json(ROOT / "evidence/n46_vs_distributed_stats.json", comparison)
    save_json(ROOT / "subsets/manifest.json", result)
    print(json.dumps({"status": "PASS", "manifest_sha256": sha256(ROOT / "subsets/manifest.json"),
                      "subsets": [{k: s[k] for k in ["replicate", "condition", "frame_count", "stats_sha256"]} for s in result["subsets"]],
                      "n46_matches_distributed": comparison["matches_distributed_stats"]}, indent=2))

if __name__ == "__main__":
    main()
