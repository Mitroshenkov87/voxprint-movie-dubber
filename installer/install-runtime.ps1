<#
  Voxprint AI Movie Dubber - online part of the installer.
  Creates a private Python 3.11 environment in <AppDir>\runtime and installs PyTorch and the other dependencies into it.
  Nothing is installed system-wide.  Safe to run again (an existing environment is updated).
    -AppDir        installation folder
    -Requirements  requirements.txt (everything except torch)
    -Backend       PyTorch build: auto (matches the NVIDIA driver, CPU when there is no GPU), cpu, cu128, ...
    -Log           log file (all output is copied there)
  The package manager "uv" (a single exe from the Astral GitHub release, SHA-256 checked) downloads the Python build and the wheels.
#>
param(
    [Parameter(Mandatory = $true)][string]$AppDir,
    [Parameter(Mandatory = $true)][string]$Requirements,
    [string]$Backend = "auto",
    [string]$Log = (Join-Path $env:TEMP "VoxprintMovieDubber-setup.log")
)
$env:PSModulePath = "$env:ProgramFiles\WindowsPowerShell\Modules;$env:SystemRoot\system32\WindowsPowerShell\v1.0\Modules"   # a parent PowerShell 7 session can leave a path that hides the built-in modules
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
if ($Backend -notmatch '^[a-z0-9]+$') { throw "Invalid PyTorch backend: $Backend" }

function Say([string]$text) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $text
    Write-Host $line
    Add-Content -LiteralPath $Log -Value $line -Encoding UTF8
}

function Run([string]$what, [string]$exe, [string[]]$arguments) {
    Say "$what"
    $ErrorActionPreference = "Continue"      # uv writes its progress to stderr; that is not an error
    & $exe @arguments 2>&1 | ForEach-Object { $s = "$_"; Write-Host $s; Add-Content -LiteralPath $Log -Value $s -Encoding UTF8 }
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

try {
    New-Item -ItemType Directory -Force -Path $AppDir | Out-Null
    Say "Voxprint AI Movie Dubber: installing the Python environment into $AppDir\runtime (PyTorch backend: $Backend)"

    # 1. uv
    $uvDir = Join-Path $AppDir "tools\uv"
    $uv = Join-Path $uvDir "uv.exe"
    if (-not (Test-Path $uv)) {
        New-Item -ItemType Directory -Force -Path $uvDir | Out-Null
        $zip = Join-Path $uvDir "uv.zip"
        $url = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
        Say "Downloading uv (package manager)"
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip
        $sumText = (Invoke-WebRequest -UseBasicParsing -Uri "$url.sha256").Content
        if ($sumText -is [byte[]]) { $sumText = [Text.Encoding]::ASCII.GetString($sumText) }
        $want = ($sumText.Trim() -split '\s+')[0].ToLower()
        $sha = [Security.Cryptography.SHA256]::Create()
        $stream = [IO.File]::OpenRead($zip)
        try { $have = ([BitConverter]::ToString($sha.ComputeHash($stream)) -replace '-', '').ToLower() } finally { $stream.Dispose() }
        if ($want -ne $have) { throw "uv download is corrupt (SHA-256 mismatch)" }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [IO.Compression.ZipFile]::ExtractToDirectory($zip, $uvDir)
        Remove-Item -LiteralPath $zip -Force
        if (-not (Test-Path $uv)) { throw "uv.exe was not found in the downloaded archive" }
    }

    # 2. Python 3.11 + the environment (all inside the app folder, so the uninstaller removes it)
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $AppDir "python"
    $env:UV_CACHE_DIR = Join-Path $AppDir "tools\uv-cache"
    $env:UV_PYTHON_PREFERENCE = "only-managed"
    $env:UV_LINK_MODE = "copy"
    $env:PYTHONUTF8 = "1"
    $runtime = Join-Path $AppDir "runtime"
    $py = Join-Path $runtime "Scripts\python.exe"
    Remove-Item -LiteralPath (Join-Path $runtime ".install-complete") -Force -ErrorAction SilentlyContinue
    Run "Creating the Python 3.11 environment" $uv @("venv", $runtime, "--python", "3.11", "--allow-existing")

    # 3. PyTorch first (the build is chosen by --torch-backend), then the rest.
    #    "auto" picks the newest CUDA build the driver supports (cu130 on recent drivers).  CTranslate2 (faster-whisper) is built
    #    for CUDA 12 and needs cublas64_12.dll / cuDNN 9 for CUDA 12, which only the cu12x PyTorch builds ship -> with cu130 the
    #    speech recognition silently runs on the CPU.  The newest cu12x build of current PyTorch is cu126, which covers GPUs up to
    #    compute capability 9.x (RTX 20/30/40); Blackwell (RTX 50, 12.x) needs cu128+, so there "auto" stays (ASR then uses the CPU).
    if ($Backend -eq "auto") {
        $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($smi) {
            try {
                $q = (& $smi.Source --query-gpu=driver_version,compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
                $drv, $cap = ($q -split ',') | ForEach-Object { $_.Trim() }
                if ([double]($drv.Split('.')[0]) -ge 560 -and [double]$cap -lt 10) {
                    $Backend = "cu126"
                    Say "NVIDIA driver ${drv}, compute capability ${cap}: using the CUDA 12.6 PyTorch build (the speech recognizer needs CUDA 12)"
                }
            } catch { Say "nvidia-smi query failed; PyTorch build left to auto" }
        }
    }
    Run "Installing PyTorch ($Backend build) - this is the largest download" $uv @("pip", "install", "--python", $py, "--compile-bytecode", "torch", "torchaudio", "--torch-backend=$Backend")
    Run "Installing the other packages" $uv @("pip", "install", "--python", $py, "--compile-bytecode", "-r", $Requirements)
    Run "Checking the installation" $py @("-c", "import torch, transformers, PySide6, faster_qwen3_tts; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")

    # 4. LGPL ffmpeg + ffprobe (BtbN FFmpeg-Builds, release 8.1 line, SHA-256 checked against the release's checksums.sha256)
    #    into <AppDir>\bin, where dubber.ffmpeg looks first.  Not fatal: without it the GPL imageio-ffmpeg fallback still works.
    $binDir = Join-Path $AppDir "bin"
    if (-not (Test-Path (Join-Path $binDir "ffprobe.exe"))) {
        try {
            $base = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
            $name = "ffmpeg-n8.1-latest-win64-lgpl-8.1.zip"
            $tmpZip = Join-Path $env:TEMP $name
            Say "Downloading ffmpeg (LGPL build, about 170 MB)"
            Invoke-WebRequest -UseBasicParsing -Uri "$base/$name" -OutFile $tmpZip
            $sums = (Invoke-WebRequest -UseBasicParsing -Uri "$base/checksums.sha256").Content
            if ($sums -is [byte[]]) { $sums = [Text.Encoding]::ASCII.GetString($sums) }
            $line = ($sums -split "`n") | Where-Object { $_ -match [regex]::Escape($name) } | Select-Object -First 1
            if (-not $line) { throw "no checksum for $name" }
            $want = ($line.Trim() -split '\s+')[0].ToLower()
            $have = (Get-FileHash -LiteralPath $tmpZip -Algorithm SHA256).Hash.ToLower()
            if ($want -ne $have) { throw "ffmpeg download is corrupt (SHA-256 mismatch)" }
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            New-Item -ItemType Directory -Force -Path $binDir | Out-Null
            $zipObj = [IO.Compression.ZipFile]::OpenRead($tmpZip)
            try {
                foreach ($e in $zipObj.Entries) {
                    if ($e.FullName -match '/bin/(ffmpeg|ffprobe)\.exe$') {
                        [IO.Compression.ZipFileExtensions]::ExtractToFile($e, (Join-Path $binDir $e.Name), $true)
                    }
                    if ($e.FullName -match '/LICENSE\.txt$') {
                        [IO.Compression.ZipFileExtensions]::ExtractToFile($e, (Join-Path $binDir "FFMPEG-LICENSE.txt"), $true)
                    }
                }
            } finally { $zipObj.Dispose() }
            Remove-Item -LiteralPath $tmpZip -Force -ErrorAction SilentlyContinue
            Say "ffmpeg (LGPL) installed into $binDir"
        } catch {
            Say ("WARNING: LGPL ffmpeg not installed (" + $_.Exception.Message + "); the bundled fallback build will be used")
        }
    }

    # 5. cleanup of the download cache; the marker is written last
    Remove-Item -LiteralPath $env:UV_CACHE_DIR -Recurse -Force -ErrorAction SilentlyContinue
    Set-Content -LiteralPath (Join-Path $runtime ".install-complete") -Value (Get-Date -Format s) -Encoding ASCII
    Say "Python environment is ready."
    exit 0
}
catch {
    Say ("ERROR: " + $_.Exception.Message)
    exit 1
}
