#!/usr/bin/env python3
"""P0の設定照合・初期状態ファイル検査・normalizerの人工データ検査。CPUのみ。"""
from pathlib import Path
import hashlib
import inspect
import json
import os

import numpy as np
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor.normalize_processor import NormalizerProcessorStep

ROOT = Path(__file__).resolve().parent
SOURCE = Path.home() / "projects/demo-d3"

def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def short(path):
    return str(path).replace(str(Path.home()), "~")

def save(name, value):
    (ROOT / "evidence" / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

def main():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    cfg = json.loads((ROOT / "experiment_conditions.p0_input.json").read_text())
    path = SOURCE / "outputs/phase2_lora_r16_s30000_46ep/checkpoints/030000/pretrained_model/train_config.json"
    old = json.loads(path.read_text())
    # 旧seedと新seedの違いは計画で意図された変更。ここでは固定表の共通条件を照合する。
    pairs = {
        "steps": (old["steps"], cfg["training"]["steps"]),
        "batch_size": (old["batch_size"], cfg["training"]["batch_size"]),
        "num_workers": (old["num_workers"], cfg["training"]["num_workers"]),
        "optimizer_lr": (old["optimizer"]["lr"], cfg["training"]["optimizer_lr"]),
        "warmup_steps": (old["scheduler"]["num_warmup_steps"], cfg["training"]["warmup_steps"]),
        "decay_steps": (old["scheduler"]["num_decay_steps"], cfg["training"]["decay_steps"]),
        "decay_lr": (old["scheduler"]["decay_lr"], cfg["training"]["decay_lr"]),
        "chunk_size": (old["policy"]["chunk_size"], cfg["training"]["chunk_size"]),
        "n_action_steps": (old["policy"]["n_action_steps"], cfg["training"]["n_action_steps"]),
        "lora_rank": (old["peft"]["r"], cfg["training"]["lora_rank"]),
        "lora_alpha": (old["peft"]["lora_alpha"], cfg["training"]["lora_alpha"]),
        "N10_N20_N46_image_transforms_enabled": (old["dataset"]["image_transforms"]["enable"], False),
    }
    settings = {"source": short(path), "source_sha256": sha256(path), "optimizer_all_values": old["optimizer"],
                "scheduler_all_values": old["scheduler"], "image_transforms_all_values": old["dataset"]["image_transforms"],
                "normalization_mapping": old["policy"]["normalization_mapping"], "peft": old["peft"],
                "matches": {k: {"old_value": a, "planned_value": b, "equal": a == b} for k, (a, b) in pairs.items()},
                "fixed_snapshot_paths_required": True,
                "legacy_revision_fields": {"dataset": old["dataset"]["revision"],
                    "policy_pretrained_revision": old["policy"]["pretrained_revision"]},
                "note": "Old config uses dataset tag v3.0 and null policy revision. P1 must load the pinned local snapshot paths explicitly. N20B brightness is a new planned transform, not the old disabled ColorJitter preset."}
    settings["all_comparable_fixed_values_match"] = all(v["equal"] for v in settings["matches"].values())
    save("settings_check.json", settings)
    # 初期状態は既知のNumPy型だけを許可し、安全なweights_only読込を維持する。
    init = SOURCE / ".venv/lib/python3.12/site-packages/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.pruned_init"
    init_sha = sha256(init)
    assert init_sha == cfg["evaluation"]["init_state_file_sha256"]
    allow = [(np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
             np.ndarray, np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType]
    with torch.serialization.safe_globals(allow):
        states = torch.load(init, weights_only=True, map_location="cpu")
    states = np.asarray(states)
    assert states.shape == (50, 92) and np.unique(states, axis=0).shape[0] == 50
    save("initial_state_check.json", {"path": short(init), "sha256": init_sha,
        "shape": list(states.shape), "dtype": str(states.dtype), "unique_rows": 50,
        "safe_loader": "torch.load(weights_only=True, map_location='cpu') with explicit NumPy allowlist",
        "row_sha256": [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in states],
        "proposed_final_state_indices": cfg["evaluation"]["proposed_final_state_indices"],
        "training_demonstration_disjointness": "NOT_ESTABLISHED"})
    module = Path(inspect.getfile(NormalizerProcessorStep)).resolve()
    assert module == (SOURCE / "lerobot/src/lerobot/processor/normalize_processor.py").resolve()
    # 第2列を定数にし、標準偏差0でもNaN/infにならず、元データへ戻せるかを検証する。
    values = torch.tensor([[1., 7.], [3., 7.], [5., 7.]], dtype=torch.float64, device="cpu")
    mean = values.mean(0); std = values.std(0, correction=0)
    features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    stats = {k: {"mean": mean.tolist(), "std": std.tolist()} for k in features}
    proc = NormalizerProcessorStep(features=features,
            norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MEAN_STD},
            stats=stats, device="cpu", dtype=torch.float64)
    outputs = {}
    for key, ft in [("observation.state", FeatureType.STATE), ("action", FeatureType.ACTION)]:
        normalized = proc._apply_transform(values, key, ft)
        restored = proc._apply_transform(normalized, key, ft, inverse=True)
        expected = (values - mean) / (std + 1e-8)
        assert torch.equal(normalized, expected)
        assert torch.isfinite(normalized).all() and torch.equal(normalized[:, 1], torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(restored, values, atol=3e-8, rtol=0)
        off_constant = torch.tensor([[3., 8.]], dtype=torch.float64)
        extrapolated = proc._apply_transform(off_constant, key, ft)
        outputs[key] = {"normalized": normalized.tolist(), "restored": restored.tolist(),
            "roundtrip_max_abs_error": float((restored - values).abs().max()),
            "off_constant_example_input": off_constant.tolist(), "off_constant_example_output": extrapolated.tolist()}
    assert not torch.cuda.is_initialized()
    save("normalizer_cpu_test.json", {"status": "PASS", "module": short(module), "module_sha256": sha256(module),
        "code_lines": [97, 349, 359, 361, 362], "input": values.tolist(), "mean": mean.tolist(),
        "population_std": std.tolist(), "eps": proc.eps, "outputs": outputs,
        "cpu_only": True, "cuda_initialized": False,
        "interpretation": "MEAN_STD uses std + 1e-8, not a std=1 replacement. Inverse uses std without epsilon. A value different from a constant training column can produce a large finite normalized value."})
    print(json.dumps({"settings_match": settings["all_comparable_fixed_values_match"], "initial_state_shape": list(states.shape),
        "initial_state_unique_rows": 50, "normalizer_test": "PASS", "cuda_initialized": False}, indent=2))

if __name__ == "__main__":
    main()
