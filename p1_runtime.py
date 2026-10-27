"""P1/P2実験用の隔離環境。既存リポジトリやキャッシュを書き換えない。"""
from pathlib import Path
import os, sys, json, hashlib
ROOT = Path(__file__).resolve().parent
SOURCE = Path.home() / "projects/demo-d3"
BASE = SOURCE / "hf_cache/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
OLD = SOURCE / "outputs/phase2_lora_r16_s30000_46ep/checkpoints/030000/pretrained_model"
BASE_SHA = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()
def short(path): return str(path).replace(str(Path.home()), "~")
def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
def setup():
    """ライブラリの一時ファイルを作業フォルダ内へ向け、逸脱した書込を拒否する。"""
    folders = {"TMPDIR":"tmp", "HF_HOME":"cache/hf", "HF_HUB_CACHE":"cache/hf/hub",
      "XDG_CACHE_HOME":"cache/xdg", "XDG_CONFIG_HOME":"cache/config", "TORCH_HOME":"cache/torch",
      "TORCH_EXTENSIONS_DIR":"cache/torch_extensions", "MPLCONFIGDIR":"cache/matplotlib",
      "NUMBA_CACHE_DIR":"cache/numba", "TRITON_CACHE_DIR":"cache/triton",
      "__GL_SHADER_DISK_CACHE_PATH":"cache/gl", "LIBERO_CONFIG_PATH":"cache/libero"}
    for key, rel in folders.items():
        p=ROOT/rel; p.mkdir(parents=True,exist_ok=True); os.environ[key]=str(p)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
      HF_HUB_DISABLE_IMPLICIT_TOKEN="1", PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled",
      MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", TOKENIZERS_PARALLELISM="false")
    sys.dont_write_bytecode=True
    if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
    os.chdir(ROOT/"dev")
    # 既存の公開モデルファイルだけを参照する。元キャッシュへの書込はaudit hookで拒否。
    for repo in ["models--lerobot--smolvla_base","models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"]:
        src=SOURCE/"hf_cache/hub"/repo
        if not src.exists(): raise RuntimeError("Required offline cache missing: "+repo)
        for f in src.rglob("*"):
            if f.is_file():
                dst=ROOT/"cache/hf/hub"/repo/f.relative_to(src)
                dst.parent.mkdir(parents=True,exist_ok=True)
                if not dst.exists():
                    try: dst.symlink_to(f.resolve())
                    except FileExistsError: pass
    import yaml
    cfg=yaml.safe_load((SOURCE/"libero_config/config.yaml").read_text())
    cfg["init_states"]=str(ROOT/"dev/init_states_only_0_9")
    savecfg=ROOT/"cache/libero/config.yaml"
    config_text=yaml.safe_dump(cfg)
    if not savecfg.exists() or savecfg.read_text()!=config_text: savecfg.write_text(config_text)
    def audit(event,args):
        if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            mode=args[1] or ""; flags=args[2] if isinstance(args[2],int) else 0
            writing=any(x in mode for x in "wax+") or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if not writing and p.suffix==".pruned_init" and p.is_relative_to(SOURCE):
                raise PermissionError("Original complete initial states are blocked in P1")
            if writing and not p.is_relative_to(ROOT) and str(p)!="/dev/null":
                raise PermissionError("Outside-workspace write blocked: "+str(p))
            if any(x in p.parts for x in (".aws",".ssh",".codex",".claude")) or p.name.startswith(".env") or p.name in ("token","stored_tokens"):
                raise PermissionError("Private file read blocked")
    sys.addaudithook(audit)
