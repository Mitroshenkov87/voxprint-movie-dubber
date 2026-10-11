<#
  Voxprint AI Movie Dubber - runtime part of the installer (online and full).
  Installs (or reuses) the shared Python runtime and links it into the program folder.  Nothing is installed system-wide.
    -AppDir        installation folder (gets the junction <AppDir>\runtime -> <runtime>\env and the uv tool)
    -Requirements  requirements.txt (everything except torch)
    -Constraints   installer\runtime-constraints.txt (pins that follow from the torch version)
    -Lock          dubber\infra\runtime_lock.json (same schema as the Audiobook Builder lock: Python 3.14, torch 2.11.0, flavor cu130)
    -Backend       auto (cu130 when the GPU passes the gate) or an explicit flavor. -Backend cpu is CI-only (GPU-less runners).
    -HardwareFile  receives the English stop message when the GPU or driver is below the minimum
    -Log           log file (all output is copied there)
    -OutFile       receives the runtime folder (one line) for the setup program
    -Payload       full (offline) installer only: folder with uv\uv.exe, python\ (the Python build + release.txt), wheels\ and
                   ffmpeg\ (installer\make-full-payload.ps1 builds it).  Nothing is downloaded when it is given.
    -SkipGpuGate   CI only: lets the full installer's cu130 runtime be installed on a GPU-less test runner.  Not offered to users.
    -ManifestScript dubber\infra\shared_manifest.py (stdlib). Records this program's reference in shared\manifest.json.
    -SkipModels    CI only: create the shared models folder and its reference, but do not treat a missing download as a failure.
                   The setup's own last step downloads the weights unless /SKIPMODELS=1 is passed.
  Rules (docs/SHARED-RESOURCES.md; dubber/infra/runtime.py has the same key logic):
    key = SHA-256 of Python + platform + torch version/flavor + requirements + constraints (12 hex digits), stored in runtime-key.json
    %LOCALAPPDATA%\Voxprint\shared\runtimes\py3.14-torch2.11-<flavor> is reused when that version is already installed;
    a shared runtime is never upgraded in place to a different Python or torch line (that line has its own directory).
    shared\manifest.json (schema 1) counts the programs using each resource. An older runtime-<12 hex digits> folder is moved once.
    <Voxprint home>\runtime belongs to the Audiobook Builder and is never moved.
  The package manager "uv" (a single exe from the Astral GitHub release, SHA-256 checked) downloads the Python build and the wheels.
  Downloads of uv and ffmpeg resume from a .partial file when a previous run was interrupted.
#>
param(
    [Parameter(Mandatory = $true)][string]$AppDir,
    [Parameter(Mandatory = $true)][string]$Requirements,
    [Parameter(Mandatory = $true)][string]$Constraints,
    [Parameter(Mandatory = $true)][string]$Lock,
    [string]$Backend = "auto",
    [string]$HardwareFile = "",
    [string]$Log = (Join-Path $env:TEMP "VoxprintMovieDubber-setup.log"),
    [string]$OutFile = "",
    [string]$Payload = "",
    [switch]$SkipGpuGate,
    [switch]$KeyOnly,
    [string]$ManifestScript = "",
    [switch]$SkipModels
)
if ($env:SystemRoot) { $env:PSModulePath = "$env:ProgramFiles\WindowsPowerShell\Modules;$env:SystemRoot\system32\WindowsPowerShell\v1.0\Modules" }   # a parent PowerShell 7 session can leave a path that hides the built-in modules
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
if ($Backend -notmatch '^[a-z0-9]+$') { throw "Invalid PyTorch backend: $Backend" }
$UserKey = "movie-dubber"

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

function Sha256Hex([string]$text) {
    $sha = [Security.Cryptography.SHA256]::Create()
    $bytes = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($text))
    return ([BitConverter]::ToString($bytes) -replace '-', '').ToLower()
}

function NormalizeReq([string]$path) {
    $out = @()
    foreach ($line in (Get-Content -LiteralPath $path -Encoding UTF8)) {
        $l = ($line -split '#', 2)[0].Trim()
        if ($l) { $out += $l }
    }
    return ($out -join "`n")
}

function RuntimeKey($lockObj, [string]$flavor) {
    $material = "python=$($lockObj.python);platform=$($lockObj.platform);torch=$($lockObj.torch_version)+$flavor;" +
                "req=$(Sha256Hex (NormalizeReq $Requirements));constraints=$(Sha256Hex (NormalizeReq $Constraints))"
    return (Sha256Hex $material).Substring(0, 12)
}

# The setup wizard has no language catalog. The dialog is this English text; the same facts are also logged in Russian.
$InstallerLead = "Voxprint AI Movie Dubber needs an NVIDIA GeForce RTX 40-series graphics card or newer (Ada Lovelace or later, compute capability 8.9 or higher) and an NVIDIA driver from the 600 branch or newer."
$InstallerLeadRu = "Для Voxprint AI Movie Dubber нужна видеокарта NVIDIA GeForce RTX 40 или новее (Ada Lovelace и новее, вычислительная способность 8.9 или выше) и драйвер NVIDIA ветки 600 или новее."

function DriverCuda {
    # "CUDA Version: 12.8" / "CUDA UMD Version: 13.4" from the nvidia-smi header = the newest CUDA the driver supports; $null without an NVIDIA GPU
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { return $null }
    try {
        $txt = (& $smi.Source 2>$null) -join "`n"
        # older drivers: "CUDA Version: 12.8"; drivers 6xx (real case, 617.42): "CUDA UMD Version: 13.4"
        if ($txt -match 'CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)') { return @([int]$Matches[1], [int]$Matches[2]) }
        # no version in the header: CUDA 13.0 (the cu130 build) needs driver branch 580 or newer.
        # The hardware gate already requires branch 600, which is above that line.
        $drv = ((& $smi.Source --query-gpu=driver_version --format=csv,noheader 2>$null) | Select-Object -First 1)
        if ($drv -match '^\s*(\d+)\.') {
            $major = [int]$Matches[1]
            if ($major -ge 580) { return @(13, 0) }
        }
    } catch { }
    return $null
}

function ChooseFlavor($lockObj, $cuda) {
    # Newest CUDA flavor the driver supports. Never "cpu": that build is only -Backend cpu, for CI.
    $best = ""; $bestV = 0
    foreach ($f in $lockObj.flavors) {
        if ($f -match '^cu(\d+)(\d)$' -and $cuda) {
            $v = [int]$Matches[1] * 10 + [int]$Matches[2]
            if ($v -le ($cuda[0] * 10 + $cuda[1]) -and $v -gt $bestV) { $best = $f; $bestV = $v }
        }
    }
    return $best
}

function LockWheelVersion($lockObj, [string]$dist, [string]$flavor) {
    foreach ($w in $lockObj.wheels) {
        if ($w.dist -eq $dist -and $w.flavor -eq $flavor) { return [string]$w.version }
    }
    return ""
}

function Convert-ComputeCap([string]$text) {
    if ($text -match '^\s*(\d+)\.(\d+)\s*$') { return @([int]$Matches[1], [int]$Matches[2]) }
    return $null
}

function Get-DriverBranch([string]$version) {
    if ($version -match '^\s*(\d+)') { return [int]$Matches[1] }
    return $null
}

function Get-PolicyGpus {
    # nvidia-smi --query-gpu=name,compute_cap,driver_version
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { return @() }
    $lines = @()
    try { $lines = @(& $smi.Source --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>$null) } catch { return @() }
    $gpus = @()
    foreach ($line in $lines) {
        if (-not $line -or ($line -notmatch ',')) { continue }
        $parts = @($line.Split(',') | ForEach-Object { $_.Trim() })
        if ($parts.Count -lt 3) { continue }
        $driver = $parts[$parts.Count - 1]
        $cap = Convert-ComputeCap $parts[$parts.Count - 2]
        $branch = Get-DriverBranch $driver
        $name = ($parts[0..($parts.Count - 3)] -join ',').Trim()
        if (-not $name -or -not $cap -or $null -eq $branch) { continue }
        $gpus += [pscustomobject]@{ Name = $name; Major = $cap[0]; Minor = $cap[1]; Driver = $driver; Branch = $branch }
    }
    return $gpus
}

function Test-GpuMeetsPolicy($gpu) {
    $capOk = ($gpu.Major -gt 8) -or (($gpu.Major -eq 8) -and ($gpu.Minor -ge 9))
    return [bool]($capOk -and ($gpu.Branch -ge 600))
}

function Format-HardwareFound($gpus) {
    $list = @($gpus)
    if ($list.Count -eq 0) { return "No NVIDIA GPU was found." }
    $bits = @()
    foreach ($g in $list) {
        $bits += "$($g.Name), compute capability $($g.Major).$($g.Minor), driver $($g.Driver) (branch $($g.Branch))"
    }
    $text = ($bits -join "; ")
    if (-not $text.EndsWith(".")) { $text += "." }
    return $text
}

function Get-InstallerStopMessage([string]$found) {
    return @"
$InstallerLead

This PC does not meet that requirement, so setup stopped and nothing was installed. A processor-only copy is not offered.

What we found: $found

Install a supported graphics card and driver, then run setup again.

NVIDIA driver downloads: https://www.nvidia.com/Download/index.aspx
"@.Trim()
}

function Get-InstallerStopMessageRu([string]$found) {
    return @"
$InstallerLeadRu

Этот компьютер не подходит, поэтому установка остановлена и ничего не установлено. Вариант только для процессора не предлагается.

Что обнаружено: $found

Установите подходящую видеокарту и драйвер и запустите установку снова.
"@.Trim()
}

function Stop-ForHardware([string]$message) {
    Say $message
    Say (Get-InstallerStopMessageRu (Format-HardwareFound @(Get-PolicyGpus)))
    if ($HardwareFile) {
        try {
            # English dialog text. UTF-8 of this ASCII message is what Inno Setup reads back as ANSI.
            [IO.File]::WriteAllText($HardwareFile, ($message + "`r`n"), (New-Object Text.UTF8Encoding $false))
        } catch { }
    }
    exit 2
}

function Resume-Download([string]$Url, [string]$Dest) {
    # Continue a dest.partial file with an HTTP Range request. A full 200 response replaces it.
    $partial = "$Dest.partial"
    $have = [int64]0
    if (Test-Path -LiteralPath $partial) { $have = (Get-Item -LiteralPath $partial).Length }
    $req = [System.Net.HttpWebRequest]::Create($Url)
    $req.UserAgent = "VoxprintMovieDubber"
    $req.Timeout = 120000
    $req.ReadWriteTimeout = 120000
    if ($have -gt 0) {
        $req.AddRange($have)
        Say "Resuming download ($have bytes already saved)"
    }
    try {
        $resp = $req.GetResponse()
    } catch {
        if ($have -gt 0 -and ("$($_.Exception.Message)" -match '416')) {
            Move-Item -LiteralPath $partial -Destination $Dest -Force
            return
        }
        throw
    }
    $status = [int]$resp.StatusCode
    $mode = if ($have -gt 0 -and $status -eq 206) { [IO.FileMode]::Append } else { [IO.FileMode]::Create }
    $fs = New-Object IO.FileStream $partial, $mode, ([IO.FileAccess]::Write)
    try {
        $src = $resp.GetResponseStream()
        $src.CopyTo($fs)
    } finally {
        $fs.Dispose()
        $resp.Dispose()
    }
    Move-Item -LiteralPath $partial -Destination $Dest -Force
}

function Add-SharedRef([string]$Id, [string]$Version, [string]$ResourcePath, [string]$Sha, [string]$Size) {
    if (-not $script:ManifestScript) { throw "-ManifestScript is required (dubber\infra\shared_manifest.py)" }
    if (-not (Test-Path -LiteralPath $script:py)) { throw "the shared Python is not available, so the manifest cannot be updated" }
    $arguments = @($script:ManifestScript, "add", "--home", $script:vxHome, "--id", $Id, "--version", $Version, "--path", $ResourcePath, "--app", $script:UserKey)
    if ($Sha) { $arguments += @("--sha256", $Sha) }
    if ($Size -ne "") { $arguments += @("--size", $Size) }
    & $script:py @arguments
    if ($LASTEXITCODE -ne 0) { throw "could not record $Id $Version in shared\manifest.json" }
}

function RemoveLinkOrDir([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return }
    $item = Get-Item -LiteralPath $path -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        [IO.Directory]::Delete($path)                 # the junction only, never what it points to
    } else {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}

$lockObj = Get-Content -LiteralPath $Lock -Raw -Encoding UTF8 | ConvertFrom-Json
if ($KeyOnly) {                                       # tests: print the key for a flavor
    RuntimeKey $lockObj $Backend
    exit 0
}

try {
    New-Item -ItemType Directory -Force -Path $AppDir | Out-Null
    $tv = $lockObj.torch_version
    if (@($lockObj.flavors) -notcontains $Backend -and $Backend -ne "auto") {
        throw "PyTorch flavor $Backend is not in the runtime lock (allowed: $($lockObj.flavors -join ', '))"
    }
    # -Backend cpu is the CI switch (.github/workflows passes /TORCH=cpu). Users are not offered a CPU install.
    if ($Payload -and -not (Test-Path (Join-Path $Payload "payload.json"))) { throw "the full installer's payload is missing: $Payload" }
    if ($Payload) { $payloadInfo = Get-Content -LiteralPath (Join-Path $Payload "payload.json") -Raw -Encoding UTF8 | ConvertFrom-Json }
    if ($Backend -eq "cpu") {
        if ($Payload) { throw "the full installer carries the $($payloadInfo.flavor) build only" }
        Say "CI switch -Backend cpu: installing the CPU build of PyTorch for an automated test machine. This build is not offered to users."
        $flavor = "cpu"
    } elseif ($SkipGpuGate) {
        if (-not $Payload) { throw "-SkipGpuGate is only for the full installer's CI test" }
        $flavor = [string]$payloadInfo.flavor
        Say "CI switch -SkipGpuGate: installing the $flavor runtime without the GPU check (automated test machine only)."
    } else {
        $policyGpus = @(Get-PolicyGpus)
        $passed = @($policyGpus | Where-Object { Test-GpuMeetsPolicy $_ })
        if ($passed.Count -eq 0) {
            Stop-ForHardware (Get-InstallerStopMessage (Format-HardwareFound $policyGpus))
        }
        $cuda = DriverCuda
        if ($Backend -eq "auto") {
            $flavor = ChooseFlavor $lockObj $cuda
            if (-not $flavor) {
                $found = Format-HardwareFound $policyGpus
                Stop-ForHardware (Get-InstallerStopMessage "$found The driver does not support CUDA 13.0, which is the only GPU build this program installs.")
            }
            Say ("NVIDIA driver supports CUDA {0}.{1}: PyTorch {2} {3}" -f $cuda[0], $cuda[1], $tv, $flavor)
            if ($Payload -and $flavor -ne [string]$payloadInfo.flavor) { throw "the full installer carries the $($payloadInfo.flavor) build; this driver needs $flavor" }
        } else {
            $flavor = $Backend
            Say "PyTorch $tv $flavor"
        }
    }
    # VRAM tier (suite rule): logged here; the program picks the tier again at every start (dubber/infra/vram_tier.py)
    $smiCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smiCmd) {
        try {
            $mib = [double](((& $smiCmd.Source --query-gpu=memory.total --format=csv,noheader,nounits 2>$null) | Select-Object -First 1).Trim())
            $tier = if ($mib / 1024 -ge 22) { "24gb" } else { "16gb" }
            Say ("GPU memory {0:N1} GB: VRAM tier {1}" -f ($mib / 1024), $tier)
        } catch { Say "GPU memory: unknown (VRAM tier is picked when the program starts)" }
    }
    $key = RuntimeKey $lockObj $flavor
    $vxHome = if ($env:VOXPRINT_HOME) { $env:VOXPRINT_HOME } else { Join-Path $env:LOCALAPPDATA "Voxprint" }
    $shared = Join-Path $vxHome "shared"
    $rtName = "py3.14-torch2.11-$flavor"
    $rt = Join-Path $shared "runtimes\$rtName"
    $envDir = Join-Path $rt "env"
    $py = Join-Path $envDir "Scripts\python.exe"
    $check = @("-c", "import torch, transformers, PySide6, faster_qwen3_tts, faster_whisper; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")
    Say "[1/5] Voxprint AI Movie Dubber: runtime $rtName (Python $($lockObj.python), torch $tv+$flavor) -> $rt"
    # An older runtime-<12 hex digits> folder moves into this versioned directory once. <home>\runtime is the Audiobook Builder's and is not touched.
    if (-not (Test-Path (Join-Path $rt "runtime-key.json"))) {
        foreach ($old in (Get-ChildItem -LiteralPath $vxHome -Directory -Filter "runtime-*" -ErrorAction SilentlyContinue)) {
            if ($old.Name -notmatch '^runtime-[0-9a-f]{12}$' -or -not (Test-Path (Join-Path $old.FullName "runtime-key.json"))) { continue }
            try { $oldInfo = Get-Content -LiteralPath (Join-Path $old.FullName "runtime-key.json") -Raw -Encoding UTF8 | ConvertFrom-Json } catch { continue }
            $oldFlavor = [string]$oldInfo.flavor
            if (-not $oldFlavor -and ([string]$oldInfo.torch).EndsWith("+cpu")) { $oldFlavor = "cpu" }
            if (-not $oldFlavor) { $oldFlavor = "cu130" }
            if ($oldFlavor -ne $flavor) { continue }
            Say "Moving $($old.FullName) into $rt"
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $rt) | Out-Null
            if (Test-Path -LiteralPath $rt) { Remove-Item -LiteralPath $rt -Recurse -Force -ErrorAction SilentlyContinue }
            Move-Item -LiteralPath $old.FullName -Destination $rt
            break
        }
    }

    # 1. uv (kept in the program folder)
    $uvDir = Join-Path $AppDir "tools\uv"
    $uv = Join-Path $uvDir "uv.exe"
    if ($Payload) {
        New-Item -ItemType Directory -Force -Path $uvDir | Out-Null
        Copy-Item -LiteralPath (Join-Path $Payload "uv\uv.exe") -Destination $uv -Force      # the uv that built the payload
    }
    if (-not (Test-Path $uv)) {
        New-Item -ItemType Directory -Force -Path $uvDir | Out-Null
        $zip = Join-Path $uvDir "uv.zip"
        $url = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
        Say "[1/5] Downloading uv (package manager)"
        Resume-Download $url $zip
        $sumText = (Invoke-WebRequest -UseBasicParsing -Uri "$url.sha256").Content
        if ($sumText -is [byte[]]) { $sumText = [Text.Encoding]::ASCII.GetString($sumText) }
        $want = ($sumText.Trim() -split '\s+')[0].ToLower()
        $have = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLower()
        if ($want -ne $have) { throw "uv download is corrupt (SHA-256 mismatch)" }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [IO.Compression.ZipFile]::ExtractToDirectory($zip, $uvDir)
        Remove-Item -LiteralPath $zip -Force
        if (-not (Test-Path $uv)) { throw "uv.exe was not found in the downloaded archive" }
    }

    # 2. reuse on an exact key match (a broken copy is repaired with the SAME pins - never upgraded)
    $reuse = $false
    if ((Test-Path (Join-Path $rt ".install-complete")) -and (Test-Path $py)) {
        $ErrorActionPreference = "Continue"
        & $py @check 2>&1 | ForEach-Object { Add-Content -LiteralPath $Log -Value "$_" -Encoding UTF8 }
        $reuse = ($LASTEXITCODE -eq 0)
        $ErrorActionPreference = "Stop"
        if ($reuse) { Say "Reusing the shared runtime $rt (same pinned versions)" } else { Say "The shared runtime $rt failed its check: repairing it with the same versions" }
    }
    if (-not $reuse) {
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $rt "python"
        $env:UV_CACHE_DIR = Join-Path $AppDir "tools\uv-cache"
        $env:UV_PYTHON_PREFERENCE = "only-managed"
        $env:UV_LINK_MODE = "copy"
        $env:PYTHONUTF8 = "1"
        New-Item -ItemType Directory -Force -Path $rt | Out-Null
        Remove-Item -LiteralPath (Join-Path $rt ".install-complete") -Force -ErrorAction SilentlyContinue
        $offline = @()
        if ($Payload) {
            # uv reads the Python build from a local mirror laid out like the python-build-standalone releases: <mirror>\<release>\<file>
            $pyDir = Join-Path $Payload "python"
            $release = (Get-Content -LiteralPath (Join-Path $pyDir "release.txt") -Encoding UTF8 | Select-Object -First 1).Trim()
            $mirror = Join-Path $env:TEMP "vmd-python-mirror"
            New-Item -ItemType Directory -Force -Path (Join-Path $mirror $release) | Out-Null
            Get-ChildItem -LiteralPath $pyDir -Filter "cpython-*" | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $mirror $release) -Force }
            $env:UV_PYTHON_INSTALL_MIRROR = "file:///" + ($mirror -replace '\\', '/')
            $offline = @("--offline", "--no-index", "--find-links", (Join-Path $Payload "wheels"))
            Say "Full installer: Python, PyTorch and the other packages come from the installer itself (no download)"
        }
        Run "[2/5] Creating the Python $($lockObj.python) environment" $uv @("venv", $envDir, "--python", $lockObj.python, "--allow-existing")
        $taVer = LockWheelVersion $lockObj "torchaudio" $flavor
        $tcVer = LockWheelVersion $lockObj "torchcodec" $flavor
        if (-not $taVer -or -not $tcVer) { throw "runtime lock has no torchaudio/torchcodec wheel for flavor $flavor" }
        $taBase = ($taVer -split '\+', 2)[0]
        $tcBase = ($tcVer -split '\+', 2)[0]
        $pins = Join-Path $env:TEMP "vmd-constraints-$key.txt"
        # exact local builds (2.11.0+cu130, torchaudio 2.11.0+cu130, torchcodec 0.17.0+cu130) so the second step keeps them
        $pinText = "torch==$tv+$flavor`r`ntorchaudio==$taVer`r`ntorchcodec==$tcVer`r`n" + ((Get-Content -LiteralPath $Constraints -Encoding UTF8) -join "`r`n")
        [IO.File]::WriteAllText($pins, $pinText, (New-Object Text.UTF8Encoding $false))
        if ($Payload) {
            Run "[2/5] Installing PyTorch $tv ($flavor build), torchaudio $taBase and torchcodec $tcBase" $uv (@("pip", "install", "--python", $py, "--compile-bytecode", "torch==$tv+$flavor", "torchaudio==$taVer", "torchcodec==$tcVer") + $offline)
        } else {
            Run "[2/5] Installing PyTorch $tv ($flavor build), torchaudio $taBase and torchcodec $tcBase" $uv @("pip", "install", "--python", $py, "--compile-bytecode", "torch==$tv", "torchaudio==$taBase", "torchcodec==$tcBase", "--torch-backend=$flavor")
        }
        Run "[3/5] Installing the other packages" $uv (@("pip", "install", "--python", $py, "--compile-bytecode", "-r", $Requirements, "-c", $pins) + $offline)
        # no --torch-backend on this step: it would re-resolve every requirement against the PyTorch index
        Run "Checking the installation" $py $check
        $info = [ordered]@{ key = $key; python = $lockObj.python; platform = $lockObj.platform; torch = "$tv+$flavor"; flavor = $flavor
                            requirements = (Sha256Hex (NormalizeReq $Requirements)); constraints = (Sha256Hex (NormalizeReq $Constraints))
                            lock_generated = $lockObj.generated; created_by = $UserKey; created = (Get-Date -Format s) }
        [IO.File]::WriteAllText((Join-Path $rt "runtime-key.json"), ($info | ConvertTo-Json), (New-Object Text.UTF8Encoding $false))
        Set-Content -LiteralPath (Join-Path $rt ".install-complete") -Value (Get-Date -Format s) -Encoding ASCII
        Remove-Item -LiteralPath $env:UV_CACHE_DIR -Recurse -Force -ErrorAction SilentlyContinue
        if ($Payload) { Remove-Item -LiteralPath (Join-Path $env:TEMP "vmd-python-mirror") -Recurse -Force -ErrorAction SilentlyContinue }
    }

    # 3. link the runtime into the program folder and drop leftover runtime-<12 hex digits> directories
    $link = Join-Path $AppDir "runtime"
    RemoveLinkOrDir $link                                     # older builds had a private environment here
    RemoveLinkOrDir (Join-Path $AppDir "python")              # ... and its base Python
    New-Item -ItemType Junction -Path $link -Target $envDir | Out-Null
    if (-not (Test-Path (Join-Path $link "Scripts\python.exe"))) { throw "the runtime link $link does not work" }
    Say "[3/5] Linked $link -> $envDir"
    [IO.File]::WriteAllText((Join-Path $AppDir "runtime-dir.txt"), $rt, (New-Object Text.UTF8Encoding $false))   # the uninstaller reads it
    foreach ($old in (Get-ChildItem -LiteralPath $vxHome -Directory -Filter "runtime-*" -ErrorAction SilentlyContinue)) {
        if ($old.Name -notmatch '^runtime-[0-9a-f]{12}$') { continue }
        Say "Removing the old runtime $($old.FullName)"
        Remove-Item -LiteralPath $old.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
    if ($OutFile) { Set-Content -LiteralPath $OutFile -Value $rt -Encoding UTF8 }

    # 4. LGPL ffmpeg + ffprobe (BtbN FFmpeg-Builds, release 8.1) into shared\ffmpeg\n8.1. A failure stops the setup.
    $ffDir = Join-Path $shared "ffmpeg\n8.1"
    $oldBin = Join-Path $AppDir "bin"
    if ((Test-Path (Join-Path $oldBin "ffmpeg.exe")) -and -not (Test-Path (Join-Path $ffDir "ffmpeg.exe"))) {
        New-Item -ItemType Directory -Force -Path $ffDir | Out-Null
        foreach ($n in @("ffmpeg.exe", "ffprobe.exe", "FFMPEG-LICENSE.txt")) {
            $src = Join-Path $oldBin $n
            if (Test-Path -LiteralPath $src) { Move-Item -LiteralPath $src -Destination (Join-Path $ffDir $n) -Force }
        }
    }
    $ffExe = Join-Path $ffDir "ffmpeg.exe"
    $ffOk = $false
    if (Test-Path -LiteralPath $ffExe) {
        $ErrorActionPreference = "Continue"
        & $ffExe -version 2>&1 | Out-Null
        $ffOk = ($LASTEXITCODE -eq 0)
        $ErrorActionPreference = "Stop"
    }
    if (-not $ffOk) {
        $base = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
        $name = "ffmpeg-n8.1-latest-win64-lgpl-8.1.zip"
        $tmpZip = Join-Path $env:TEMP $name
        if ($Payload) {
            Say "[4/5] Installing ffmpeg (LGPL build) from the installer"
            Copy-Item -LiteralPath (Join-Path $Payload "ffmpeg\$name") -Destination $tmpZip -Force
            $sums = Get-Content -LiteralPath (Join-Path $Payload "ffmpeg\checksums.sha256") -Raw -Encoding ASCII
        } else {
            Say "[4/5] Downloading ffmpeg (LGPL build, about 170 MB)"
            Resume-Download "$base/$name" $tmpZip
            $sums = (Invoke-WebRequest -UseBasicParsing -Uri "$base/checksums.sha256").Content
        }
        if ($sums -is [byte[]]) { $sums = [Text.Encoding]::ASCII.GetString($sums) }
        $line = ($sums -split "`n") | Where-Object { $_ -match [regex]::Escape($name) } | Select-Object -First 1
        if (-not $line) { throw "no checksum for $name" }
        $want = ($line.Trim() -split '\s+')[0].ToLower()
        $have = (Get-FileHash -LiteralPath $tmpZip -Algorithm SHA256).Hash.ToLower()
        if ($want -ne $have) {
            Remove-Item -LiteralPath $tmpZip -Force -ErrorAction SilentlyContinue
            throw "ffmpeg download is corrupt (SHA-256 mismatch)"
        }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        New-Item -ItemType Directory -Force -Path $ffDir | Out-Null
        $zipObj = [IO.Compression.ZipFile]::OpenRead($tmpZip)
        try {
            foreach ($e in $zipObj.Entries) {
                if ($e.FullName -match '/bin/(ffmpeg|ffprobe)\.exe$') {
                    [IO.Compression.ZipFileExtensions]::ExtractToFile($e, (Join-Path $ffDir $e.Name), $true)
                }
                if ($e.FullName -match '/LICENSE\.txt$') {
                    [IO.Compression.ZipFileExtensions]::ExtractToFile($e, (Join-Path $ffDir "FFMPEG-LICENSE.txt"), $true)
                }
            }
        } finally { $zipObj.Dispose() }
        Remove-Item -LiteralPath $tmpZip -Force -ErrorAction SilentlyContinue
    }
    foreach ($n in @("ffmpeg.exe", "ffprobe.exe", "FFMPEG-LICENSE.txt")) {
        $src = Join-Path $oldBin $n
        if (Test-Path -LiteralPath $src) { Remove-Item -LiteralPath $src -Force -ErrorAction SilentlyContinue }
    }
    if (-not (Test-Path -LiteralPath $ffExe)) { throw "ffmpeg was not installed into $ffDir" }
    $ErrorActionPreference = "Continue"
    & $ffExe -version 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { $ErrorActionPreference = "Stop"; throw "ffmpeg was extracted but does not run" }
    $ErrorActionPreference = "Stop"
    Say "[4/5] ffmpeg (LGPL) installed into $ffDir"

    # 5. shared models folder and the manifest references. The setup downloads the weights unless -SkipModels.
    $models = Join-Path $shared "models"
    New-Item -ItemType Directory -Force -Path $models | Out-Null
    if ($SkipModels) {
        Say "[5/5] CI switch -SkipModels: the shared models folder is ready and the weights are not downloaded here"
    } else {
        Say "[5/5] Shared models folder $models (the setup downloads the default-pipeline models next)"
    }
    Add-SharedRef "runtime" $rtName $rt "" ""
    Add-SharedRef "torch" "$tv+$flavor" $rt "" ""
    Add-SharedRef "ctranslate2" "cu12" $rt "" ""
    $ffItem = Get-Item -LiteralPath $ffExe
    Add-SharedRef "ffmpeg" "n8.1" $ffDir ((Get-FileHash -LiteralPath $ffExe -Algorithm SHA256).Hash.ToLower()) $ffItem.Length.ToString()
    Add-SharedRef "models" "store" $models "" ""

    Say "Python environment is ready."
    exit 0
}
catch {
    Say ("ERROR: " + $_.Exception.Message)
    exit 1
}
