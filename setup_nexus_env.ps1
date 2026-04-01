$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $RepoRoot '.venv'
$PythonExe = Join-Path $VenvDir 'Scripts\python.exe'
$BootstrapPython = $env:NEXUS_INSTANTMESH_BOOTSTRAP_PYTHON

function Invoke-Step {
    param(
        [Parameter(Mandatory)]
        [string]$Description,
        [Parameter(Mandatory)]
        [scriptblock]$Action
    )

    Write-Host $Description -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Description (exit code $LASTEXITCODE)"
    }
}

function Resolve-BootstrapPython {
    if ($BootstrapPython) {
        return $BootstrapPython
    }
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        return 'py -3.10'
    }
    return 'python'
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw 'InstantMesh native bootstrap requires an NVIDIA CUDA runtime. nvidia-smi was not found on this host.'
}

$BootstrapCommand = Resolve-BootstrapPython
Write-Host "Preparing InstantMesh NEXUS environment in $VenvDir" -ForegroundColor Cyan
Write-Host "Bootstrap python command: $BootstrapCommand" -ForegroundColor DarkGray

if (-not (Test-Path $PythonExe)) {
    Invoke-Step "Creating Python 3.10 venv" {
        if ($BootstrapCommand -eq 'py -3.10') {
            py -3.10 -m venv $VenvDir
        }
        else {
            & $BootstrapCommand -m venv $VenvDir
        }
    }
}

Invoke-Step "Upgrading pip tooling" {
    & $PythonExe -m pip install --upgrade pip setuptools wheel
}

Invoke-Step "Installing CUDA PyTorch 2.1.0 toolchain" {
    & $PythonExe -m pip install `
        torch==2.1.0 `
        torchvision==0.16.0 `
        torchaudio==2.1.0 `
        --index-url https://download.pytorch.org/whl/cu121
}

Invoke-Step "Installing xformers and core inference dependencies" {
    & $PythonExe -m pip install `
        xformers==0.0.22.post7 `
        ninja `
        pytorch-lightning==2.1.2 `
        huggingface-hub `
        einops `
        omegaconf `
        torchmetrics `
        webdataset `
        accelerate `
        tensorboard `
        PyMCubes `
        trimesh `
        rembg `
        transformers==4.34.1 `
        diffusers==0.20.2 `
        imageio[ffmpeg] `
        xatlas `
        plyfile `
        gradio==3.41.2
}

Push-Location $RepoRoot
try {
    Invoke-Step "Installing nvdiffrast" {
        & $PythonExe -m pip install git+https://github.com/NVlabs/nvdiffrast/
    }
}
finally {
    Pop-Location
}

Write-Host "InstantMesh NEXUS environment bootstrap complete." -ForegroundColor Green
Write-Host "Backend python: $PythonExe" -ForegroundColor Gray
Write-Host "Next: place TencentARC/InstantMesh checkpoints under .\\ckpts or allow public download at runtime." -ForegroundColor Gray
