<#
  Voxprint AI Movie Dubber - runtime part of the installer (online and full).
  Installs (or reuses) the shared Python runtime and links it into the program folder.  Nothing is installed system-wide.
    -AppDir        installation folder (gets the junction <AppDir>\runtime -> <runtime>\env, the uv tool and the LGPL ffmpeg)
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
  Rules agreed with Voxprint AI Audiobook Builder (dubber/infra/runtime.py has the same key logic):
    key = SHA-256 of Python + platform + torch version/flavor + requirements + constraints (12 hex digits)
    %LOCALAPPDATA%\Voxprint\runtime-<key> is reused only on an exact key match, otherwise a new folder is installed side by side;
    a shared runtime is never upgraded in place; runtime-<key>\.users.json counts the programs using it.
  The package manager "uv" (a single exe from the Astral GitHub release, SHA-256 checked) downloads the Python build and the wheels.
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
    [switch]$KeyOnly
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

function ReadUsers([string]$file) {
    $h = [ordered]@{}
    if (Test-Path -LiteralPath $file) {
        try {
            $o = Get-Content -LiteralPath $file -Raw -Encoding UTF8 | ConvertFrom-Json
            foreach ($p in $o.PSObject.Properties) { $h[$p.Name] = [bool]$p.Value }
        } catch { }
    }
    return $h
}

function WriteUsers([string]$file, $h) {
    $tmp = "$file.$([guid]::NewGuid().ToString('N').Substring(0, 8)).tmp"
    $json = if ($h.Count) { ($h | ConvertTo-Json) } else { "{}" }
    [IO.File]::WriteAllText($tmp, $json, (New-Object Text.UTF8Encoding $false))
    Move-Item -LiteralPath $tmp -Destination $file -Force
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
    $rt = Join-Path $vxHome "runtime-$key"
    $envDir = Join-Path $rt "env"
    $py = Join-Path $envDir "Scripts\python.exe"
    $check = @("-c", "import torch, transformers, PySide6, faster_qwen3_tts, faster_whisper; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")
    Say "Voxprint AI Movie Dubber: runtime key $key (Python $($lockObj.python), torch $tv+$flavor) -> $rt"

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
        Say "Downloading uv (package manager)"
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip
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
        Run "Creating the Python $($lockObj.python) environment" $uv @("venv", $envDir, "--python", $lockObj.python, "--allow-existing")
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
            Run "Installing PyTorch $tv ($flavor build), torchaudio $taBase and torchcodec $tcBase" $uv (@("pip", "install", "--python", $py, "--compile-bytecode", "torch==$tv+$flavor", "torchaudio==$taVer", "torchcodec==$tcVer") + $offline)
        } else {
            Run "Installing PyTorch $tv ($flavor build), torchaudio $taBase and torchcodec $tcBase" $uv @("pip", "install", "--python", $py, "--compile-bytecode", "torch==$tv", "torchaudio==$taBase", "torchcodec==$tcBase", "--torch-backend=$flavor")
        }
        Run "Installing the other packages" $uv (@("pip", "install", "--python", $py, "--compile-bytecode", "-r", $Requirements, "-c", $pins) + $offline)
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

    # 3. register this program as a user of the runtime; link it into the program folder
    $usersFile = Join-Path $rt ".users.json"
    $u = ReadUsers $usersFile
    $u[$UserKey] = $true
    WriteUsers $usersFile $u
    $link = Join-Path $AppDir "runtime"
    RemoveLinkOrDir $link                                     # older builds had a private environment here
    RemoveLinkOrDir (Join-Path $AppDir "python")              # ... and its base Python
    New-Item -ItemType Junction -Path $link -Target $envDir | Out-Null
    if (-not (Test-Path (Join-Path $link "Scripts\python.exe"))) { throw "the runtime link $link does not work" }
    Say "Linked $link -> $envDir"
    [IO.File]::WriteAllText((Join-Path $AppDir "runtime-dir.txt"), $rt, (New-Object Text.UTF8Encoding $false))   # the uninstaller reads it

    # 4. runtimes this program used before (other keys): drop our key; delete the folder when nobody else uses it
    foreach ($old in (Get-ChildItem -LiteralPath $vxHome -Directory -Filter "runtime-*" -ErrorAction SilentlyContinue)) {
        if ($old.FullName -eq $rt -or $old.Name -notmatch '^runtime-[0-9a-f]{12}$' -or -not (Test-Path (Join-Path $old.FullName "runtime-key.json"))) { continue }
        $f = Join-Path $old.FullName ".users.json"
        $ou = ReadUsers $f
        if (-not $ou.Contains($UserKey)) { continue }
        $ou.Remove($UserKey)
        WriteUsers $f $ou
        if (@($ou.Keys | Where-Object { $ou[$_] }).Count -eq 0) {
            Say "Removing the old runtime $($old.FullName) (no program uses it any more)"
            Remove-Item -LiteralPath $old.FullName -Recurse -Force -ErrorAction SilentlyContinue
        } else {
            Say "Old runtime $($old.FullName) kept: still used by $(@($ou.Keys) -join ', ')"
        }
    }
    if ($OutFile) { Set-Content -LiteralPath $OutFile -Value $rt -Encoding UTF8 }

    # 5. LGPL ffmpeg + ffprobe (BtbN FFmpeg-Builds, release 8.1 line, SHA-256 checked against the release's checksums.sha256)
    #    into <AppDir>\bin, where dubber.ffmpeg looks first.  Not fatal: without it the GPL imageio-ffmpeg fallback still works.
    $binDir = Join-Path $AppDir "bin"
    if (-not (Test-Path (Join-Path $binDir "ffprobe.exe"))) {
        try {
            $base = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
            $name = "ffmpeg-n8.1-latest-win64-lgpl-8.1.zip"
            $tmpZip = Join-Path $env:TEMP $name
            if ($Payload) {
                Say "Installing ffmpeg (LGPL build) from the installer"
                Copy-Item -LiteralPath (Join-Path $Payload "ffmpeg\$name") -Destination $tmpZip -Force
                $sums = Get-Content -LiteralPath (Join-Path $Payload "ffmpeg\checksums.sha256") -Raw -Encoding ASCII
            } else {
                Say "Downloading ffmpeg (LGPL build, about 170 MB)"
                Invoke-WebRequest -UseBasicParsing -Uri "$base/$name" -OutFile $tmpZip
                $sums = (Invoke-WebRequest -UseBasicParsing -Uri "$base/checksums.sha256").Content
            }
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

    Say "Python environment is ready."
    exit 0
}
catch {
    Say ("ERROR: " + $_.Exception.Message)
    exit 1
}
