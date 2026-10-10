<#
  Voxprint AI Movie Dubber - builds the payload of the FULL (offline) installer on a Windows build machine.
    -Out           payload folder (created; must be empty or missing)
    -Requirements  requirements.txt
    -Constraints   installer\runtime-constraints.txt
    -Lock          dubber\infra\runtime_lock.json (Python 3.14, torch 2.11.0, flavor cu130)
    -Python        a Python 3.14 interpreter for "pip wheel" (the build machine's, not shipped)
    -Flavor        the PyTorch build to bundle (default cu130; the only GPU build the program installs)
  Layout read by install-runtime.ps1 -Payload:
    payload.json          flavor, Python release, versions (what this payload carries)
    uv\uv.exe             the uv that the setup uses (same binary that picked the Python build below)
    python\<build>.tar.gz python-build-standalone archive for Windows x64; python\release.txt = its release tag
    wheels\*.whl          every wheel the runtime needs: torch/torchaudio/torchcodec (+flavor) and requirements.txt
    ffmpeg\<zip>          the LGPL ffmpeg build install-runtime.ps1 would download, with checksums.sha256
  The AI models are not part of the payload: they are downloaded on first use (or by the setup's optional models task).
#>
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [Parameter(Mandatory = $true)][string]$Requirements,
    [Parameter(Mandatory = $true)][string]$Constraints,
    [Parameter(Mandatory = $true)][string]$Lock,
    [Parameter(Mandatory = $true)][string]$Python,
    [string]$Flavor = "cu130"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Fetch([string]$url, [string]$file) {
    Write-Host "download: $url"
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $file
}

function Check([string]$what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

$lockObj = Get-Content -LiteralPath $Lock -Raw -Encoding UTF8 | ConvertFrom-Json
if (@($lockObj.flavors) -notcontains $Flavor -or $Flavor -eq "cpu") { throw "flavor $Flavor is not a GPU flavor of the runtime lock" }
$tv = [string]$lockObj.torch_version
$taVer = ""; $tcVer = ""
foreach ($w in $lockObj.wheels) {
    if ($w.dist -eq "torchaudio" -and $w.flavor -eq $Flavor) { $taVer = [string]$w.version }
    if ($w.dist -eq "torchcodec" -and $w.flavor -eq $Flavor) { $tcVer = [string]$w.version }
}
if (-not $taVer -or -not $tcVer) { throw "runtime lock has no torchaudio/torchcodec wheel for $Flavor" }

if ((Test-Path $Out) -and (Get-ChildItem -LiteralPath $Out -Force | Select-Object -First 1)) { throw "payload folder is not empty: $Out" }
foreach ($d in "uv", "python", "wheels", "ffmpeg") { New-Item -ItemType Directory -Force -Path (Join-Path $Out $d) | Out-Null }

# 1. uv (SHA-256 checked against the release's .sha256)
$uvUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
$uvZip = Join-Path $env:TEMP "uv-payload.zip"
Fetch $uvUrl $uvZip
$sumText = (Invoke-WebRequest -UseBasicParsing -Uri "$uvUrl.sha256").Content
if ($sumText -is [byte[]]) { $sumText = [Text.Encoding]::ASCII.GetString($sumText) }
if ((($sumText.Trim() -split '\s+')[0].ToLower()) -ne (Get-FileHash -LiteralPath $uvZip -Algorithm SHA256).Hash.ToLower()) { throw "uv download is corrupt" }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipObj = [IO.Compression.ZipFile]::OpenRead($uvZip)
try {
    $entry = $zipObj.Entries | Where-Object { $_.Name -eq "uv.exe" } | Select-Object -First 1
    if (-not $entry) { throw "uv.exe is not in the uv archive" }
    [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $Out "uv\uv.exe"), $true)
} finally { $zipObj.Dispose() }
$uv = Join-Path $Out "uv\uv.exe"
$uvVersion = (& $uv --version).Trim()
Write-Host "uv: $uvVersion"

# 2. the Python build this uv installs for "--python <lock python>" (same binary at setup time = same choice)
$downloads = (& $uv python list $lockObj.python --only-downloads --output-format json) | ConvertFrom-Json
Check "uv python list"
$pick = $downloads | Where-Object { $_.os -eq "windows" -and "$($_.arch)" -match "x86_64" -and $_.variant -eq "default" -and $_.implementation -eq "cpython" } |
    Sort-Object { [version]$_.version } -Descending | Select-Object -First 1
if (-not $pick) { throw "uv lists no CPython $($lockObj.python) build for Windows x64" }
$segments = ([Uri]$pick.url).Segments
$release = $segments[$segments.Count - 2].TrimEnd('/')
$pyFile = [Uri]::UnescapeDataString($segments[$segments.Count - 1])
Fetch $pick.url (Join-Path $Out "python\$pyFile")
[IO.File]::WriteAllText((Join-Path $Out "python\release.txt"), $release, (New-Object Text.UTF8Encoding $false))
Write-Host "python: $($pick.version) ($release, $pyFile)"

# 3. wheels: the same pins install-runtime.ps1 uses (exact local builds of torch, torchaudio and torchcodec)
$pins = Join-Path $env:TEMP "vmd-payload-constraints.txt"
$pinText = "torch==$tv+$Flavor`r`ntorchaudio==$taVer`r`ntorchcodec==$tcVer`r`n" + ((Get-Content -LiteralPath $Constraints -Encoding UTF8) -join "`r`n")
[IO.File]::WriteAllText($pins, $pinText, (New-Object Text.UTF8Encoding $false))
$wheels = Join-Path $Out "wheels"
& $Python -m pip wheel --disable-pip-version-check --wheel-dir $wheels --extra-index-url "https://download.pytorch.org/whl/$Flavor" `
    -c $pins "torch==$tv+$Flavor" "torchaudio==$taVer" "torchcodec==$tcVer" -r $Requirements
Check "pip wheel"
$count = @(Get-ChildItem -LiteralPath $wheels -Filter "*.whl").Count
$torchWheel = Get-ChildItem -LiteralPath $wheels -Filter "torch-$tv+$Flavor-*.whl" | Select-Object -First 1
if (-not $torchWheel) { throw "the torch $tv+$Flavor wheel is missing from the payload" }
Write-Host "wheels: $count"

# 4. ffmpeg (LGPL, the build install-runtime.ps1 downloads), SHA-256 checked
$base = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
$name = "ffmpeg-n8.1-latest-win64-lgpl-8.1.zip"
Fetch "$base/$name" (Join-Path $Out "ffmpeg\$name")
Fetch "$base/checksums.sha256" (Join-Path $Out "ffmpeg\checksums.sha256")
$line = (Get-Content -LiteralPath (Join-Path $Out "ffmpeg\checksums.sha256")) | Where-Object { $_ -match [regex]::Escape($name) } | Select-Object -First 1
if (-not $line -or (($line.Trim() -split '\s+')[0].ToLower() -ne (Get-FileHash -LiteralPath (Join-Path $Out "ffmpeg\$name") -Algorithm SHA256).Hash.ToLower())) { throw "ffmpeg download is corrupt" }

$info = [ordered]@{ flavor = $Flavor; python = [string]$pick.version; python_release = $release; torch = "$tv+$Flavor"
                    torchaudio = $taVer; torchcodec = $tcVer; uv = $uvVersion; wheels = $count; built = (Get-Date).ToUniversalTime().ToString("s") + "Z" }
[IO.File]::WriteAllText((Join-Path $Out "payload.json"), ($info | ConvertTo-Json), (New-Object Text.UTF8Encoding $false))
Get-Content (Join-Path $Out "payload.json")
"payload size: {0:N2} GB" -f ((Get-ChildItem -LiteralPath $Out -Recurse -File | Measure-Object Length -Sum).Sum / 1GB)
