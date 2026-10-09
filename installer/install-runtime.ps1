<#
  Voxprint AI Movie Dubber - online part of the installer.
  Installs (or reuses) the shared Python runtime and links it into the program folder.  Nothing is installed system-wide.
    -AppDir        installation folder (gets the junction <AppDir>\runtime -> <runtime>\env, the uv tool and the LGPL ffmpeg)
    -Requirements  requirements.txt (everything except torch)
    -Constraints   installer\runtime-constraints.txt (pins that follow from the torch version)
    -Lock          dubber\infra\runtime_lock.json (verbatim copy of the Audiobook Builder's lock: torch version, flavors, Python)
    -Backend       PyTorch flavor: auto (newest flavor of the lock the NVIDIA driver supports, CPU without a GPU), cpu, cu126, cu128
    -Log           log file (all output is copied there)
    -OutFile       receives the runtime folder (one line) for the setup program
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
    [string]$Log = (Join-Path $env:TEMP "VoxprintMovieDubber-setup.log"),
    [string]$OutFile = "",
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

function DriverCuda {
    # "CUDA Version: 12.8" / "CUDA UMD Version: 13.4" from the nvidia-smi header = the newest CUDA the driver supports; $null without an NVIDIA GPU
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { return $null }
    try {
        $txt = (& $smi.Source 2>$null) -join "`n"
        # older drivers: "CUDA Version: 12.8"; drivers 6xx (real case, 617.42): "CUDA UMD Version: 13.4"
        if ($txt -match 'CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)') { return @([int]$Matches[1], [int]$Matches[2]) }
        # no version in the header: the driver number tells the minimum (570+ runs CUDA 12.8, 560+ 12.6)
        $drv = ((& $smi.Source --query-gpu=driver_version --format=csv,noheader 2>$null) | Select-Object -First 1)
        if ($drv -match '^\s*(\d+)\.') {
            $major = [int]$Matches[1]
            if ($major -ge 570) { return @(12, 8) }
            if ($major -ge 560) { return @(12, 6) }
        }
    } catch { }
    return $null
}

function ChooseFlavor($lockObj, $cuda) {
    $best = "cpu"; $bestV = 0
    foreach ($f in $lockObj.flavors) {
        if ($f -match '^cu(\d+)(\d)$' -and $cuda) {
            $v = [int]$Matches[1] * 10 + [int]$Matches[2]
            if ($v -le ($cuda[0] * 10 + $cuda[1]) -and $v -gt $bestV) { $best = $f; $bestV = $v }
        }
    }
    return $best
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
    $cuda = DriverCuda
    if ($Backend -eq "auto") {
        $flavor = ChooseFlavor $lockObj $cuda
        if ($cuda) { Say ("NVIDIA driver supports CUDA {0}.{1}: PyTorch {2} {3}" -f $cuda[0], $cuda[1], $tv, $flavor) } else { Say "No NVIDIA GPU found: PyTorch $tv cpu" }
    } else {
        if (@($lockObj.flavors) -notcontains $Backend) { throw "PyTorch flavor $Backend is not in the runtime lock (allowed: $($lockObj.flavors -join ', '))" }
        $flavor = $Backend
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
        Run "Creating the Python $($lockObj.python) environment" $uv @("venv", $envDir, "--python", $lockObj.python, "--allow-existing")
        $pins = Join-Path $env:TEMP "vmd-constraints-$key.txt"
        $pinText = "torch==$tv`r`ntorchaudio==$tv`r`n" + ((Get-Content -LiteralPath $Constraints -Encoding UTF8) -join "`r`n")
        [IO.File]::WriteAllText($pins, $pinText, (New-Object Text.UTF8Encoding $false))
        Run "Installing PyTorch $tv ($flavor build) - this is the largest download" $uv @("pip", "install", "--python", $py, "--compile-bytecode", "torch==$tv", "torchaudio==$tv", "--torch-backend=$flavor")
        Run "Installing the other packages" $uv @("pip", "install", "--python", $py, "--compile-bytecode", "-r", $Requirements, "-c", $pins, "--torch-backend=$flavor")
        Run "Checking the installation" $py $check
        $info = [ordered]@{ key = $key; python = $lockObj.python; platform = $lockObj.platform; torch = "$tv+$flavor"; flavor = $flavor
                            requirements = (Sha256Hex (NormalizeReq $Requirements)); constraints = (Sha256Hex (NormalizeReq $Constraints))
                            lock_generated = $lockObj.generated; created_by = $UserKey; created = (Get-Date -Format s) }
        [IO.File]::WriteAllText((Join-Path $rt "runtime-key.json"), ($info | ConvertTo-Json), (New-Object Text.UTF8Encoding $false))
        Set-Content -LiteralPath (Join-Path $rt ".install-complete") -Value (Get-Date -Format s) -Encoding ASCII
        Remove-Item -LiteralPath $env:UV_CACHE_DIR -Recurse -Force -ErrorAction SilentlyContinue
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

    Say "Python environment is ready."
    exit 0
}
catch {
    Say ("ERROR: " + $_.Exception.Message)
    exit 1
}
