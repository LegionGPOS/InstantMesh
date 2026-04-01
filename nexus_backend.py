#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
CODEX_ROOT = REPO_ROOT.parents[1]
if str(CODEX_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEX_ROOT))

RUN_SCRIPT = REPO_ROOT / "run.py"
DEFAULT_CONFIG = os.environ.get(
    "NEXUS_INSTANTMESH_CONFIG",
    str(REPO_ROOT / "configs" / "instant-mesh-large.yaml"),
)
DEFAULT_RUNTIME_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _force_cuda_state() -> bool | None:
    forced = str(os.environ.get("NEXUS_FORCE_CUDA_STATE", "")).strip().lower()
    if forced in {"1", "true", "yes", "present"}:
        return True
    if forced in {"0", "false", "no", "absent"}:
        return False
    return None


def _has_cuda_runtime() -> bool:
    forced = _force_cuda_state()
    if forced is not None:
        return forced
    if shutil.which("nvidia-smi") is not None:
        return True
    cuda_path = str(os.environ.get("CUDA_PATH", "")).strip()
    if cuda_path and Path(cuda_path).exists():
        return True
    for key in ("CUDA_HOME", "CUDA_ROOT"):
        candidate = str(os.environ.get(key, "")).strip()
        if candidate and Path(candidate).exists():
            return True
    return False


def _path_has_checkpoint_payload(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    for marker in ("config.json", "model_index.json"):
        if (path / marker).exists():
            return True
    for pattern in ("*.bin", "*.ckpt", "*.pt", "*.pth", "*.safetensors"):
        if next(path.rglob(pattern), None) is not None:
            return True
    return False


def _config_name(config_path: Path) -> str:
    return config_path.stem


def _expected_model_ckpt_name(config_path: Path) -> str:
    return f"{_config_name(config_path).replace('-', '_')}.ckpt"


def _discover_local_checkpoint_root(config_path: Path, requested_root: str | None) -> Path | None:
    candidates: list[Path] = []
    if requested_root:
        candidates.append(Path(requested_root))
    for key in ("NEXUS_INSTANTMESH_CKPTS", "NEXUS_INSTANTMESH_MODEL"):
        explicit = str(os.environ.get(key, "")).strip()
        if explicit:
            candidates.append(Path(explicit))
    codex_root = REPO_ROOT.parents[1]
    candidates.extend(
        [
            REPO_ROOT / "ckpts",
            REPO_ROOT,
            codex_root / "models" / "media" / "instantmesh" / "ckpts",
            codex_root / "models" / "media" / "instantmesh",
        ]
    )
    required_files = {
        "diffusion_pytorch_model.bin",
        _expected_model_ckpt_name(config_path),
    }
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if not _path_has_checkpoint_payload(candidate):
            continue
        root = candidate
        if candidate.is_dir() and candidate.name.lower() != "ckpts":
            maybe_ckpts = candidate / "ckpts"
            if maybe_ckpts.exists():
                root = maybe_ckpts
        if root.is_dir() and all((root / name).exists() for name in required_files):
            return root.resolve()
    return None


def _resolve_runtime_python() -> Path:
    runtime_python = str(os.environ.get("NEXUS_INSTANTMESH_PYTHON", "")).strip()
    if runtime_python:
        return Path(runtime_python)
    if DEFAULT_RUNTIME_PYTHON.exists():
        return DEFAULT_RUNTIME_PYTHON
    return Path(sys.executable)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NEXUS wrapper for the official InstantMesh backend"
    )
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--backend-id", default="instantmesh", help="Backend identifier")
    parser.add_argument("--target-height", type=float, default=1.8)
    parser.add_argument("--depth-ratio", type=float, default=0.22)
    parser.add_argument("--material-finish", default="realistic")
    parser.add_argument("--body-style", default="realistic")
    parser.add_argument(
        "--device",
        default=os.environ.get("NEXUS_INSTANTMESH_DEVICE", "cuda"),
        help="InstantMesh device override. Native execution currently requires cuda.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Config path, defaulting to instant-mesh-large",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=os.environ.get("NEXUS_INSTANTMESH_CKPTS", ""),
        help="Optional local ckpts directory override",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=int(os.environ.get("NEXUS_INSTANTMESH_DIFFUSION_STEPS", "75")),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("NEXUS_INSTANTMESH_SEED", "42")),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=float(os.environ.get("NEXUS_INSTANTMESH_SCALE", "1.0")),
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=float(os.environ.get("NEXUS_INSTANTMESH_DISTANCE", "4.5")),
    )
    parser.add_argument(
        "--view",
        type=int,
        choices=(4, 6),
        default=int(os.environ.get("NEXUS_INSTANTMESH_VIEW_COUNT", "6")),
    )
    parser.add_argument(
        "--no-remove-bg",
        action="store_true",
        default=_env_flag("NEXUS_INSTANTMESH_NO_REMOVE_BG", False),
    )
    parser.add_argument(
        "--export-texmap",
        dest="export_texmap",
        action="store_true",
        default=_env_flag("NEXUS_INSTANTMESH_EXPORT_TEXMAP", True),
    )
    parser.add_argument(
        "--no-export-texmap",
        dest="export_texmap",
        action="store_false",
    )
    parser.add_argument(
        "--save-video",
        dest="save_video",
        action="store_true",
        default=_env_flag("NEXUS_INSTANTMESH_SAVE_VIDEO", False),
    )
    parser.add_argument(
        "--no-save-video",
        dest="save_video",
        action="store_false",
    )
    return parser.parse_args()


def _resolve_config_path(config_value: str) -> Path:
    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    return config_path


def _write_config_override(config_path: Path, checkpoint_root: Path, output_dir: Path) -> Path:
    config_text = config_path.read_text(encoding="utf-8")
    override_lines: list[str] = []
    unet_path = checkpoint_root / "diffusion_pytorch_model.bin"
    model_path = checkpoint_root / _expected_model_ckpt_name(config_path)
    for line in config_text.splitlines():
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]
        if stripped.startswith("unet_path:"):
            override_lines.append(f"{indent}unet_path: {json.dumps(str(unet_path))}")
            continue
        if stripped.startswith("model_path:"):
            override_lines.append(f"{indent}model_path: {json.dumps(str(model_path))}")
            continue
        override_lines.append(line)
    override_path = output_dir / f"{config_path.stem}.nexus-override.yaml"
    override_path.write_text("\n".join(override_lines) + "\n", encoding="utf-8")
    return override_path


def _build_command(
    args: argparse.Namespace,
    config_path: Path,
    output_dir: Path,
) -> tuple[list[str], Path]:
    python_bin = _resolve_runtime_python()
    command = [
        str(python_bin),
        str(RUN_SCRIPT),
        str(config_path),
        str(Path(args.input).resolve()),
        "--output_path",
        str(output_dir),
        "--diffusion_steps",
        str(args.diffusion_steps),
        "--seed",
        str(args.seed),
        "--scale",
        str(args.scale),
        "--distance",
        str(args.distance),
        "--view",
        str(args.view),
    ]
    if args.no_remove_bg:
        command.append("--no_rembg")
    if args.export_texmap:
        command.append("--export_texmap")
    if args.save_video:
        command.append("--save_video")
    return command, python_bin


def _detect_primary_asset(output_dir: Path) -> Path:
    for suffix in (".obj", ".gltf", ".glb"):
        for asset in sorted(output_dir.rglob(f"*{suffix}")):
            return asset.resolve()
    raise RuntimeError(f"InstantMesh did not produce a mesh asset in {output_dir}")


def _detect_optional_asset(output_dir: Path, suffix: str) -> str | None:
    for asset in sorted(output_dir.rglob(f"*{suffix}")):
        return str(asset.resolve())
    return None


def _write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    command: list[str],
    runtime_python: Path,
    config_path: Path,
    asset_path: Path,
    image_path: str | None,
    video_path: str | None,
    checkpoint_root: Path | None,
    completed: subprocess.CompletedProcess[str],
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    manifest_path = output_dir / "nexus-output.json"
    payload: dict[str, Any] = {
        "backend_id": args.backend_id,
        "mesh_path": str(asset_path),
        "image_path": image_path,
        "video_path": video_path,
        "metadata": {
            "native_backend": "instantmesh",
            "native_wrapper": "nexus_backend.py",
            "runtime_python": str(runtime_python),
            "config_path": str(config_path),
            "config_name": config_path.stem,
            "model_source": "local_ckpts" if checkpoint_root is not None else "huggingface_public",
            "checkpoint_root": str(checkpoint_root) if checkpoint_root is not None else None,
            "device": args.device,
            "target_height": args.target_height,
            "depth_ratio": args.depth_ratio,
            "material_finish": args.material_finish,
            "body_style": args.body_style,
            "diffusion_steps": args.diffusion_steps,
            "seed": args.seed,
            "scale": args.scale,
            "distance": args.distance,
            "view_count": args.view,
            "export_texmap": args.export_texmap,
            "save_video": args.save_video,
            "remove_background": not args.no_remove_bg,
            "native_command": command,
            "stdout_tail": completed.stdout[-1200:] if completed.stdout else "",
            "stderr_tail": completed.stderr[-1200:] if completed.stderr else "",
        },
    }
    if extra_metadata:
        payload["metadata"].update(extra_metadata)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _run_amd_compatible_emulation(
    args: argparse.Namespace,
    output_dir: Path,
    config_path: Path,
    runtime_python: Path,
    *,
    attempted_native_command: list[str] | None = None,
    attempted_completed: subprocess.CompletedProcess[str] | None = None,
) -> int:
    from nexus.media.character_rigging_3d import reconstruct_multiview_emulated_character_mesh, write_obj
    from nexus.media.multiview_gen import generate_canonical_multiviews

    config_name = config_path.stem
    source_path = Path(args.input).resolve()
    image_dir = output_dir / config_name / "images"
    mesh_dir = output_dir / config_name / "meshes"
    view_dir = output_dir / config_name / "views"
    image_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    view_dir.mkdir(parents=True, exist_ok=True)

    copied_image_path = image_dir / source_path.name
    if copied_image_path.resolve() != source_path:
        shutil.copy2(source_path, copied_image_path)

    multiview_manifest = generate_canonical_multiviews(source_path, view_dir, stem=source_path.stem)
    multiview_payload = json.loads(multiview_manifest.read_text(encoding="utf-8"))
    summary = multiview_payload.get("summary", {})
    mesh = reconstruct_multiview_emulated_character_mesh(
        source_path,
        multiview_manifest,
        target_height=float(args.target_height),
        depth_ratio=float(args.depth_ratio),
        material_finish=args.material_finish,
        body_style=args.body_style,
    )
    multiview_emulation = dict(mesh.metadata.get("multiview_emulation", {}))
    emulated_depth_ratio = float(multiview_emulation.get("emulated_depth_ratio", args.depth_ratio))
    emulated_target_height = float(multiview_emulation.get("emulated_target_height", args.target_height))
    mesh.metadata.update(
        {
            "execution_profile": "amd_compatible_emulation",
            "multiview_manifest": str(multiview_manifest),
            "multiview_summary": dict(summary),
            "emulated_depth_ratio": round(emulated_depth_ratio, 4),
            "emulated_target_height": round(emulated_target_height, 4),
        }
    )
    mesh_path = mesh_dir / f"{source_path.stem}.obj"
    write_obj(mesh, mesh_path)

    emulation_command = [
        str(runtime_python),
        "NEXUS_AMD_INSTANTMESH_EMULATION",
        str(source_path),
        str(output_dir),
    ]
    completed = subprocess.CompletedProcess(
        args=emulation_command,
        returncode=0,
        stdout="NEXUS AMD-compatible InstantMesh emulation completed successfully.",
        stderr="",
    )
    _write_manifest(
        output_dir,
        args,
        emulation_command,
        runtime_python,
        config_path,
        mesh_path,
        str(copied_image_path),
        None,
        None,
        completed,
        extra_metadata={
            "execution_profile": "amd_compatible_emulation",
            "native_cuda_available": False,
            "wrapper_native_attempted": attempted_native_command is not None,
            "wrapper_native_fallback_used": attempted_native_command is not None,
            "wrapper_native_command": attempted_native_command,
            "wrapper_native_stdout_tail": attempted_completed.stdout[-1200:] if attempted_completed and attempted_completed.stdout else "",
            "wrapper_native_stderr_tail": attempted_completed.stderr[-1200:] if attempted_completed and attempted_completed.stderr else "",
            "wrapper_native_returncode": attempted_completed.returncode if attempted_completed is not None else None,
            "wrapper_native_failure": (
                (attempted_completed.stderr or attempted_completed.stdout or "").strip()[:400]
                if attempted_completed is not None and attempted_completed.returncode != 0
                else None
            ),
            "nexus_multiview_manifest": str(multiview_manifest),
            "rendered_view_count": len(multiview_payload.get("views", [])),
            "emulated_depth_ratio": round(emulated_depth_ratio, 4),
            "emulated_target_height": round(emulated_target_height, 4),
            "model_source": "amd_compatible_emulation",
        },
    )
    return 0


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not RUN_SCRIPT.exists():
        raise FileNotFoundError(f"InstantMesh run.py not found at {RUN_SCRIPT}")

    config_path = _resolve_config_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"InstantMesh config not found at {config_path}")

    runtime_python = _resolve_runtime_python()
    if not str(args.device).strip().lower().startswith("cuda") or not _has_cuda_runtime():
        return _run_amd_compatible_emulation(
            args,
            output_dir,
            config_path,
            runtime_python,
        )

    checkpoint_root = _discover_local_checkpoint_root(config_path, args.checkpoint_root)
    effective_config = (
        _write_config_override(config_path, checkpoint_root, output_dir)
        if checkpoint_root is not None
        else config_path
    )
    command, runtime_python = _build_command(args, effective_config, output_dir)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return _run_amd_compatible_emulation(
            args,
            output_dir,
            config_path,
            runtime_python,
            attempted_native_command=command,
            attempted_completed=completed,
        )

    asset_path = _detect_primary_asset(output_dir)
    image_path = _detect_optional_asset(output_dir, ".png")
    video_path = _detect_optional_asset(output_dir, ".mp4") if args.save_video else None
    _write_manifest(
        output_dir,
        args,
        command,
        runtime_python,
        effective_config,
        asset_path,
        image_path,
        video_path,
        checkpoint_root,
        completed,
        extra_metadata={
            "execution_profile": "cuda_native",
            "native_cuda_available": True,
            "wrapper_native_attempted": True,
            "wrapper_native_fallback_used": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
