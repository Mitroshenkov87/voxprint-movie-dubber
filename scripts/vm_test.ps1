<#
  GPU virtual-machine check for Voxprint AI Movie Dubber.

    .\scripts\vm_test.ps1 [-Smoke] [-Full] [-Json] [-Clips DIR]

  -Smoke  dry-run JSON, selftest, a fetch-models check, and one short clip dub.
  -Full   every clip in the folder, plus one actor-blend dub, and a multi-voice
          dub when HF_TOKEN is set.
  -Json   print one JSON object on stdout. Progress stays on the information stream.

  Exit 0 pass, 1 a case failed, 2 the machine is not set up, 3 no NVIDIA GPU.
  Clips default to .\clips. See docs/TESTING-GPU.md.
#>
param(
    [switch]$Smoke,
    [switch]$Full,
    [switch]$Json,
    [string]$Clips = ""
)
$ErrorActionPreference = "Stop"
$Mode = if ($Full) { "full" } else { "smoke" }
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not $Clips) { $Clips = Join-Path $Root "clips" }
$Py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Say([string]$Text) { Write-Information $Text -InformationAction Continue }

function Fail-Setup([string]$Text) {
    Say "setup: $Text"
    if ($Json) { Write-Output (@{ mode = $Mode; ok = $false; exit_code = 2; error = $Text } | ConvertTo-Json -Compress) }
    exit 2
}

if (-not (Get-Command $Py -ErrorAction SilentlyContinue)) { Fail-Setup "python was not found" }
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    Say "no NVIDIA GPU: nvidia-smi is missing"
    if ($Json) { Write-Output (@{ mode = $Mode; ok = $false; exit_code = 3; error = "nvidia-smi is missing" } | ConvertTo-Json -Compress) }
    exit 3
}
& nvidia-smi --query-gpu=name --format=csv,noheader | Out-Null
if ($LASTEXITCODE -ne 0) {
    Say "no NVIDIA GPU: nvidia-smi did not report a device"
    if ($Json) { Write-Output (@{ mode = $Mode; ok = $false; exit_code = 3; error = "nvidia-smi reported no device" } | ConvertTo-Json -Compress) }
    exit 3
}
if (-not (Test-Path -LiteralPath $Clips)) { Fail-Setup "clips folder does not exist: $Clips" }
$videos = @(Get-ChildItem -LiteralPath $Clips -Recurse -File -Include *.mp4,*.mkv,*.webm,*.mov | Sort-Object FullName)
if ($videos.Count -eq 0) { Fail-Setup "no video files in $Clips" }
$short = $videos | Sort-Object Length | Select-Object -First 1
$work = Join-Path $env:TEMP ("vmd-vm-test-" + [guid]::NewGuid().ToString("n"))
New-Item -ItemType Directory -Force -Path $work | Out-Null
$log = Join-Path $work "run.log"
$cases = New-Object System.Collections.Generic.List[object]
$peak = 0
$sampler = $null

function Start-Vram {
    $script:peak = 0
    $script:sampler = Start-Job -ArgumentList $work -ScriptBlock {
        param($dir)
        $best = 0
        $peakFile = Join-Path $dir "vram.txt"
        while ($true) {
            $line = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1
            $used = 0
            if ($line -match '^\s*(\d+)') { $used = [int]$Matches[1] }
            if ($used -gt $best) { $best = $used; Set-Content -LiteralPath $peakFile -Value $best }
            Start-Sleep -Seconds 1
        }
    }
}

function Stop-Vram {
    if ($script:sampler) {
        Stop-Job $script:sampler -ErrorAction SilentlyContinue
        Remove-Job $script:sampler -Force -ErrorAction SilentlyContinue
        $script:sampler = $null
    }
    $file = Join-Path $work "vram.txt"
    if (Test-Path -LiteralPath $file) {
        $value = 0
        [void][int]::TryParse((Get-Content -LiteralPath $file -Raw).Trim(), [ref]$value)
        if ($value -gt $script:peak) { $script:peak = $value }
    }
}

function Add-Case([string]$Name, [int]$Code, [double]$Seconds, [string]$Note) {
    $cases.Add([pscustomobject]@{ name = $Name; ok = ($Code -eq 0); exit_code = $Code; seconds = [math]::Round($Seconds, 3); note = $Note })
}

function Invoke-Timed([string]$Name, [string[]]$Arguments, [switch]$WatchVram) {
    if ($WatchVram) { Start-Vram }
    $start = Get-Date
    & $Py @Arguments *>> $log
    $code = $LASTEXITCODE
    $seconds = ((Get-Date) - $start).TotalSeconds
    if ($WatchVram) { Stop-Vram }
    Add-Case $Name $code $seconds ""
}

function New-Project([string]$Name, [string]$Clip, [string]$Kind) {
    $folder = Join-Path $work $Name
    & $Py -c @"
from pathlib import Path
from dubber.core.project import Project
settings = {'target_lang': 'ru', 'source_lang': 'en'}
kind = '$Kind'
if kind == 'actor':
    settings['single_voice'] = {'kind': 'actor', 'id': '', 'actor_weight': 0.5}
elif kind == 'multi':
    settings['multi_voice'] = True
Project.create(Path(r'$folder'), Path(r'$Clip'), **settings)
"@
    if ($LASTEXITCODE -ne 0) { Fail-Setup "could not create the $Name project" }
    return $folder
}

Say "mode $Mode, clips $Clips, short clip $($short.FullName)"
Invoke-Timed "dry-run" @("$Root\main.py", "--dry-run", "--json")
Invoke-Timed "selftest" @("$Root\main.py", "selftest", "--json")
Invoke-Timed "fetch-models" @("$Root\main.py", "fetch-models", "--dry-run", "--json")
if ($Mode -eq "smoke") {
    Invoke-Timed "dub:$($short.Name)" @("$Root\main.py", "run-project", $short.FullName, "--json") -WatchVram
} else {
    foreach ($clip in $videos) {
        Invoke-Timed "dub:$($clip.Name)" @("$Root\main.py", "run-project", $clip.FullName, "--json") -WatchVram
    }
    $actor = New-Project "actor" $short.FullName "actor"
    Invoke-Timed "actor-blend" @("$Root\main.py", "run-project", $actor, "--json") -WatchVram
    if ($env:HF_TOKEN) {
        $multi = New-Project "multi" $short.FullName "multi"
        Invoke-Timed "multi-voice" @("$Root\main.py", "run-project", $multi, "--json") -WatchVram
    } else {
        Add-Case "multi-voice" 0 0 "skipped: HF_TOKEN is not set"
    }
}

$text = Get-Content -LiteralPath $log -Raw -ErrorAction SilentlyContinue
if (-not $text) { $text = "" }
$backend = ""
$note = ""
foreach ($line in ($text -split "`n")) {
    if ($line -match "TTS backend:\s*(.+)$") { $backend = $Matches[1].Trim() }
    if ($line -match "TTS backends skipped:" -or $line -match "CUDA Graphs") { $note = $line.Trim() }
}
$graphs = $backend.StartsWith("graphs") -or ($backend -match "CUDA Graphs")
if (-not $graphs -and -not $note) { $note = "the log did not name a TTS backend" }
$ok = $true
foreach ($case in $cases) { if (-not $case.ok) { $ok = $false } }
if (-not $graphs -and $note -eq "the log did not name a TTS backend") { $ok = $false }
$body = [ordered]@{
    mode = $Mode
    ok = $ok
    exit_code = $(if ($ok) { 0 } else { 1 })
    clips = $Clips
    cases = $cases
    tts_backend = $backend
    cuda_graphs = [bool]$graphs
    graphs_note = $note
    peak_vram_mib = $peak
}
$rendered = $body | ConvertTo-Json -Depth 6
if ($Json) { Write-Output $rendered } else { Say $rendered }
Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
if ($ok) { exit 0 } else { exit 1 }
