import os, re, json, shutil, argparse, sys
from pathlib import Path
from typing import Dict
from safetensors.torch import load_file, save_file

# Regex to drop any embedding / head tensors (handles different nesting prefixes)
DROP_RE = re.compile(r"(?:^|\.)(?:embed_tokens|lm_head)(?:\.|$)")

def should_skip(path: Path) -> bool:
    # Skip deepspeed/optimizer/trainer checkpoints
    name = path.name
    return (
        name.startswith("checkpoint-") or
        name in {".git", "wandb", "logs"} or
        (path.is_dir() and name == "runs")
    )

def clean_safetensors(src_file: Path, dst_file: Path) -> Dict[str, int]:
    sd = load_file(str(src_file))
    kept = {k: v for k, v in sd.items() if not DROP_RE.search(k)}
    removed = len(sd) - len(kept)
    if kept:
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        save_file(kept, str(dst_file), metadata=None)
    return {"removed": removed, "kept": len(kept)}

def resolve_path(p: str | Path, repo_root: Path) -> Path:
    p = Path(p)
    # If it's absolute, keep it. If relative, interpret relative to repo root.
    return p if p.is_absolute() else (repo_root / p)

def main():
    # Drive repo root = parent of this script's folder (tools/)
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Clean LoRA adapter (drop embed_tokens/lm_head tensors).")
    parser.add_argument("--src", type=str,
                        default=str(Path("LLaMA-Factory/saves/qwen2_vl-2b/pretrain")),
                        help="Adapter folder to clean (relative to Drive/ or absolute).")
    parser.add_argument("--dst", type=str, default="",
                        help="Output folder (relative to Drive/ or absolute). If empty, uses <src>_clean2.")
    args = parser.parse_args()

    src = resolve_path(args.src, repo_root)
    dst = resolve_path(args.dst, repo_root) if args.dst else (src.parent / (src.name + "_clean"))

    if not src.exists() or not src.is_dir():
        print(f"ERROR: src folder not found: {src}", file=sys.stderr)
        sys.exit(1)

    print(f"Repo root: {repo_root}")
    print(f"SRC: {src}")
    print(f"DST: {dst}")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    removed_total = 0
    kept_total = 0

    # Copy over non-weight files first (excluding checkpoints)
    for item in src.iterdir():
        if should_skip(item):
            continue
        if item.is_file():
            if item.suffix not in {".safetensors"} and item.name != "adapter_config.json":
                shutil.copy2(item, dst / item.name)

    # Patch adapter_config.json
    ac_src = src / "adapter_config.json"
    if ac_src.exists():
        with open(ac_src) as f:
            cfg = json.load(f)
        cfg["modules_to_save"] = []
        with open(dst / "adapter_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
    else:
        print("WARN: adapter_config.json not found in", src)

    # Clean all top-level safetensors (ignore checkpoint subfolders)
    for item in src.iterdir():
        if should_skip(item):
            continue
        if item.is_file() and item.suffix == ".safetensors":
            stats = clean_safetensors(item, dst / item.name)
            removed_total += stats["removed"]
            kept_total += stats["kept"]
            print(f"[clean] {item.name}: removed={stats['removed']} kept={stats['kept']}")

    print(f"Done. Cleaned adapter saved to: {dst}")
    print(f"Totals: removed={removed_total} kept={kept_total}")

if __name__ == "__main__":
    main()