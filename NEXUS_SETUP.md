# InstantMesh NEXUS Notes

This checkout is the NEXUS multiview-oriented reconstruction backend surface
for `InstantMesh`.

## Wrapper

- Entry point: `nexus_backend.py`
- Router protocol: `nexus_wrapper`
- Output contract: writes `nexus-output.json` plus the generated mesh asset

## Native Reality

The official `InstantMesh` runtime is CUDA-centric:

- `run.py` hard-selects `torch.device('cuda')`
- the repo depends on `nvdiffrast`
- the official README targets `Python 3.10`, `PyTorch 2.1.0`, and `CUDA 12.1`

Because of that, the NEXUS wrapper performs an explicit CUDA preflight and
falls back to an AMD-compatible emulation path on hosts without NVIDIA CUDA
availability instead of burning time on doomed launches.

## Bootstrap

Use the backend-local bootstrap script on a CUDA-capable host:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_nexus_env.ps1
```

The script:

- creates a Python 3.10 `.venv`
- installs the official CUDA PyTorch stack
- installs `xformers` and the main InstantMesh inference dependencies
- installs `nvdiffrast`

## Checkpoints

At runtime, the wrapper prefers local checkpoints when present and otherwise
falls back to the public Hugging Face model downloads used by the upstream
repo.

Recommended local layout:

```text
tools/instantmesh/ckpts/
  diffusion_pytorch_model.bin
  instant_mesh_large.ckpt
```

You can also point the wrapper at another checkpoint root with:

```powershell
$env:NEXUS_INSTANTMESH_CKPTS='D:\models\instantmesh\ckpts'
```

## Smoke Run

Example:

```powershell
python .\nexus_backend.py --input <image> --output-dir <output-dir>
```

On hosts without NVIDIA CUDA, the wrapper is expected to complete through the
AMD-compatible emulation path and still write `nexus-output.json`.
