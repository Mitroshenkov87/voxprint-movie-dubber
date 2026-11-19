# Compile the Windows GUI and console launchers (voxprint-movie-dubber.exe, voxprint-movie-dubber-cli.exe).
# Both start main.py through the shared runtime. The GUI launcher is a Windows subsystem program
# (pythonw.exe). The CLI launcher is a console program (python.exe) with the same entry point.
param(
    [Parameter(Mandatory = $true)][string]$OutDir,
    [Parameter(Mandatory = $true)][string]$Icon
)

$ErrorActionPreference = "Stop"
$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path -LiteralPath $csc)) {
    $csc = Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe"
}
if (-not (Test-Path -LiteralPath $csc)) {
    throw "csc.exe was not found. The Windows installers need the .NET Framework compiler."
}
if (-not (Test-Path -LiteralPath $Icon)) {
    throw "icon was not found: $Icon"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$source = Join-Path $PSScriptRoot "launcher.cs"
$gui = Join-Path $OutDir "voxprint-movie-dubber.exe"
$cli = Join-Path $OutDir "voxprint-movie-dubber-cli.exe"
& $csc /nologo /target:winexe "/win32icon:$Icon" "/out:$gui" $source
if ($LASTEXITCODE -ne 0) { throw "the GUI launcher failed to compile ($LASTEXITCODE)" }
& $csc /nologo /target:exe /define:CONSOLE "/win32icon:$Icon" "/out:$cli" $source
if ($LASTEXITCODE -ne 0) { throw "the CLI launcher failed to compile ($LASTEXITCODE)" }
"launchers: $gui"
"launchers: $cli"
