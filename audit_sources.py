#!/usr/bin/env python3
"""既存コードと保存ログを読み取り、P0の追跡根拠を保存する。環境は起動しない。"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = Path.home() / "projects/demo-d3"
REPO = SOURCE / "lerobot"
SRC = REPO / "src/lerobot"
SITE = SOURCE / ".venv/lib/python3.12/site-packages"

def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def short(path):
    return str(path).replace(str(Path.home()), "~")

def save(name, value):
    (ROOT / "evidence" / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

def main():
    # 旧ログと結果JSONが、既存出典表に記載された実行のファイルであることをhashで照合する。
    runs = []
    expected = [
        ("phase2_formal_base_30ep", "9af4d1cf2046ceddf141795136735b2925da5875e0b45f3042a8a4fe2cf868f7", "f8ed39182b549c46cfdb67ea91e42a39b46d09f98b10f2a1c8560bb5f3f2a303"),
        ("phase2_formal_adapted_030000_30ep", "63a5ad5e14670c29be51fe4ec08a83d966cc4cabff92f28d5edb0864dc721069", "3af250aca2ae4a3b7210d0277c321824b98da7f23b93ecd93f5e33137824d9d2"),
    ]
    for name, expected_log, expected_info in expected:
        log = SOURCE / "logs" / (name + ".log")
        info = SOURCE / "outputs" / name / "eval_info.json"
        assert sha256(log) == expected_log and sha256(info) == expected_info
        text = log.read_text()
        required = ["--eval.n_episodes=30", "--eval.batch_size=10", "--seed=20260824",
                    "--env.init_states=true", "--env.hard_reset=true", "async=True"]
        assert all(token in text for token in required)
        metrics = json.loads(info.read_text())["per_task"][0]["metrics"]
        assert len(metrics["successes"]) == 30
        selected_lines = [{"line": i, "text": s.replace(str(Path.home()), "~")}
                          for i, s in enumerate(text.splitlines(), 1)
                          if "Command being timed:" in s or "Making environment (batch_size=" in s]
        runs.append({"run": name, "log": short(log), "log_sha256": expected_log,
                     "eval_info": short(info), "eval_info_sha256": expected_info,
                     "source_lines": selected_lines, "per_trial_init_state_id_recorded": False,
                     "success_count_historical": sum(metrics["successes"]), "episodes": 30})
    sources = Path.home() / "demo_videos/d3/d3_sources.md"
    source_text = sources.read_text()
    assert "7427f31801f944f5ec9ce53cda02862ce7d1638b" in source_text
    assert "7427f31801f944f5ec9ce53cda02862ce7d1638b" in (SOURCE / "repro/pip-freeze.txt").read_text()
    # コード経路からの静的な算術追跡。シミュレーションやmock環境を実行するものではない。
    batches = [{"batch": b + 1, "init_state_ids": list(range(b * 10, (b + 1) * 10)),
                "env_seeds": list(range(20260824 + b * 10, 20260824 + (b + 1) * 10))}
               for b in range(3)]
    trace = {
        "conclusion": "a",
        "statement": "The two named legacy formal runs used initial state indices 0 through 29; reconstructed from the pinned source, matching saved run hashes and recorded arguments.",
        "method": "Static deterministic code-path reconstruction; no simulator execution and no per-trial init_state_id log.",
        "scope": "Only the two named 30-trial formal runs; not a claim that indices 30-49 were never used in any other historical activity.",
        "batches": batches, "legacy_runs": runs,
        "steps": [
            {"fact": "Each environment starts at episode_index i=0..9; reset stride is n_envs=10.",
             "source": "lerobot/src/lerobot/envs/libero.py", "lines": "165-175,419-438"},
            {"fact": "Temporary metadata environment is closed without reset; async worker wrappers absorb autoresets after episode end.",
             "source": "lerobot/src/lerobot/envs/utils.py", "lines": "217-245,275-289"},
            {"fact": "LiberoEnv.reset selects state[id % 50], then increments id by 10. Internal initial raw environment reset does not advance this counter.",
             "source": "lerobot/src/lerobot/envs/libero.py", "lines": "258-277,339-352"},
            {"fact": "Three batches; each rollout explicitly resets with NEW_ROLLOUT_OPTION. Only this reset thaws the finished environment.",
             "source": "lerobot/src/lerobot/scripts/lerobot_eval.py", "lines": "219-223,470-472,519-543"},
            {"fact": "Step returns terminal observation; it does not reset the environment.",
             "source": "lerobot/src/lerobot/envs/libero.py", "lines": "366-391"},
            {"fact": "Recorded command is 30 episodes, 10 async environments, fixed init states, hard reset, seed 20260824.",
             "source": "logs/phase2_formal_base_30ep.log", "lines": "115,2333"},
            {"fact": "Same recorded execution parameters for the adapted run.",
             "source": "logs/phase2_formal_adapted_030000_30ep.log", "lines": "125,2362"}
        ],
        "historical_source_document": {"path": short(sources), "sha256": sha256(sources), "lines": "22,283-292,350-353"},
        "final_state_indices": list(range(30, 50)),
        "final_label": "本計画の開発・モデル選定に使用しない固定評価条件",
        "training_demonstration_disjointness": "NOT_ESTABLISHED",
        "non_overlap_proposal_only": "If needed later, compare original LIBERO demonstration initial simulator states with pruned_init rows using a documented state representation. No such comparison or source-data download was performed."
    }
    save("initial_state_usage_trace.json", trace)
    feasibility = [
        ("explicit_state_index", "envs/libero.py:124,165-175,339-346,419-438", "Pass episode_index=30..39 or 40..49 to explicitly constructed LiberoEnv factories. Default _make_env_fns starts at zero, so seed-only or the unmodified CLI does not implement the final-state selection."),
        ("fresh_environment_per_model_condition_wave", "envs/libero.py:393-400,419-438,445-456; envs/utils.py:312-315", "Build and close a new vector environment for each wave/model/image condition; do not reuse reset counters."),
        ("hard_reset", "envs/libero.py:131,164,267-274; envs/configs.py:415-423", "Forward hard_reset=True to the underlying environment."),
        ("settling_10_steps", "envs/libero.py:127,162,348-352", "num_steps_wait=10 executes dummy actions after state assignment."),
        ("20_hz", "envs/libero.py:128,163,267-274; envs/configs.py:325,421", "control_freq=20. Dataset metadata fps=10 is a separate property; preserve the planned evaluation frequency."),
        ("280_step_limit", "envs/libero.py:98-103,188-193; scripts/lerobot_eval.py:263,272", "Spatial task limit is 280; loop exits on all done or limit."),
        ("success_check", "envs/libero.py:374-384; installed libero/libero/envs/env_wrapper.py:103-104", "check_success delegates to LIBERO _check_success; retain the saved success flags and avoid redefining success."),
        ("policy_seed", "scripts/lerobot_eval.py:748; utils/random_utils.py:171-183; scripts/lerobot_eval.py:528-543", "set_seed controls Python/NumPy/torch RNG; environment seed list is separate. P1 must reseed after model construction and before each paired wave, with identical ordering."),
        ("save_all_videos", "scripts/lerobot_eval.py:485-493,590-612,786", "The helper supports max_episodes_rendered; CLI defaults to 10. For each fresh 10-environment wave set 10 and use a distinct directory; save 20 trials across two waves. Correct render_fps=80 metadata to the planned 20 Hz timebase in the custom recorder."),
        ("brightness_eval_hook", "scripts/lerobot_eval.py:274,289-293; envs/utils.py:99-102; policies/smolvla/processor_smolvla.py:65-76", "Insert transformation on the two observation.images keys after env_preprocessor and before policy preprocessor. Inputs are float32 RGB [0,1]. Leave state/action/task unchanged."),
        ("brightness_train_hook", "scripts/lerobot_train.py:128-139; policies/smolvla/processor_smolvla.py:65-76", "Insert after uint8-to-float conversion/rename and before preprocessor; or add one saved ProcessorStep before tokenization/device/normalization. Reuse the path during evaluation. Two-camera factor equality and separate augmentation RNG require P1 implementation/tests.")
    ]
    save("evaluator_feasibility.json", {"status": "STATIC_REVIEW_COMPLETE_NOT_EXECUTED",
        "items": [{"requirement": name, "source_lines": lines, "implementation": meaning} for name, lines, meaning in feasibility]})
    dist = SITE / "hf_libero-0.1.4.dist-info"
    license_file = dist / "licenses/LICENSE"; metadata = dist / "METADATA"
    assert "MIT License" in license_file.read_text()
    assert "Creative Commons Attribution 4.0 International (CC BY 4.0)" in metadata.read_text()
    assets = SITE / "libero/libero/assets"
    notices = [short(p) for root in [SITE / "libero", SOURCE / "libero_assets"] for p in root.rglob("*")
               if p.is_file() and any(x in p.name.lower() for x in ["license", "notice", "copying"])]
    save("license_check.json", {
        "checked_date_jst": "2026-10-10", "method": "Installed files and pinned local source; no external upload or new remote license query.",
        "libero_distribution": "hf_libero 0.1.4",
        "installed_code": {"license": "MIT", "path": short(license_file), "sha256": sha256(license_file),
                           "lines": "1-21", "copyright": "Copyright (c) 2023 Lifelong Robot Learning"},
        "installed_dataset_statement": {"license": "CC-BY-4.0", "path": short(metadata), "sha256": sha256(metadata), "lines": "229-233"},
        "matches_plan_code_and_data_labels": True,
        "asset_directories": sorted(p.name for p in assets.iterdir() if p.is_dir() and not p.name.startswith(".")),
        "additional_asset_license_files_found": notices,
        "individual_mesh_texture_rights": "UNCONFIRMED: package-level code/data labels do not establish the provenance and redistribution terms of every bundled mesh or texture. Do not redistribute bundled assets.",
        "hf_transformed_dataset_relationship": "UNCONFIRMED: plan records Apache-2.0 card versus upstream CC-BY-4.0; no additional evidence resolving that relationship.",
        "smolvla_base_and_lora_redistribution": "UNCONFIRMED; do not publish weights. Cached snapshot contains only model/config/processor files and supplies no new license evidence.",
        "new_license_differences": "No contradiction to the plan's MIT/CC-BY-4.0 labels. Installed distribution identified as hf_libero 0.1.4 (Hugging Face fork). Individual bundled asset terms remain unconfirmed.",
        "public_package": "Original scripts/necessary patches, fixed source references, settings, episode lists, summary CSVs and reproduction instructions; not weights, datasets, cached assets, private logs, or evidence files.",
        "third_party_code_copied_into_p0_helpers": False,
        "notices_if_copied_later": [
            {"source": "LeRobot", "required_notice_reference": "lerobot/LICENSE:1-201; relevant file headers; applicable bundled notices at 204 onward", "notice": "Copyright 2024 The Hugging Face team; reviewed source headers Copyright 2025 The HuggingFace Inc. team; Apache-2.0"},
            {"source": "LIBERO", "required_notice_reference": short(license_file) + ":1-21", "notice": "Copyright (c) 2023 Lifelong Robot Learning; MIT permission notice"}
        ]
    })
    paths = [sources, SOURCE / "repro/pip-freeze.txt", REPO / "LICENSE", license_file, metadata,
             SITE / "libero/libero/envs/env_wrapper.py"]
    paths += [SRC / name for name in ["envs/libero.py", "envs/utils.py", "envs/configs.py", "envs/factory.py",
        "scripts/lerobot_eval.py", "scripts/lerobot_train.py", "policies/smolvla/processor_smolvla.py",
        "processor/normalize_processor.py", "utils/random_utils.py"]]
    paths += [SOURCE / "logs" / (name + ".log") for name, _, _ in expected]
    paths += [SOURCE / "outputs" / name / "eval_info.json" for name, _, _ in expected]
    save("source_evidence_hashes.json", {"files": [{"path": short(p), "sha256": sha256(p)} for p in paths]})
    git = ["git", "--no-optional-locks", "-C", str(REPO)]
    head = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(git + ["status", "--porcelain=v1", "--untracked-files=all"], text=True)
    assert head == "7427f31801f944f5ec9ce53cda02862ce7d1638b" and status == ""
    save("preflight.json", {"plan_md5_checked_on_mac": "6ccbb83f0f1449bcce0f1fd94556b81b",
        "head": head, "git_status_porcelain": status, "free_bytes": shutil.disk_usage(ROOT).free,
        "free_gib": shutil.disk_usage(ROOT).free / 2**30, "minimum_free_gib": 50, "pass": True,
        "remote_write_root": "~/projects/pai-final-smolvla", "executed_phase": "P0_CPU_ONLY"})
    print("PASS: saved source trace, evaluator feasibility, licenses and source hashes; no simulation executed.")

if __name__ == "__main__":
    main()
