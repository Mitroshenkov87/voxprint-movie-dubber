; Voxprint AI Movie Dubber - ONLINE and FULL installers (Inno Setup 6.x).
;   ISCC installer\VoxprintMovieDubber.iss                 ->   installer\Output\VoxprintMovieDubber-Setup.exe (online)
;   ISCC /DFull installer\VoxprintMovieDubber.iss          ->   installer\Output\VoxprintMovieDubber-Full-Setup.exe + -N.bin parts
;   The full (offline) build carries the payload made by installer\make-full-payload.ps1 (default folder build\payload, or
;   /DPayloadDir=<folder>): uv, the Python 3.14 build, every wheel incl. PyTorch 2.11.0+cu130, and the LGPL ffmpeg.  The setup
;   then downloads nothing except the AI models.  It is split into parts below 2 GB (GitHub release asset limit);
;   keep all parts in one folder and start the .exe.
;
; The installer is small: it contains only the program's own files (Python sources, icon, licences) and a download script.
; While it installs, install-runtime.ps1 installs (or reuses) the shared runtime
; LOCALAPPDATA\Voxprint\shared\runtimes\py3.14-torch2.11-cu130 (Python 3.14, PyTorch 2.11.0+cu130,
; torchaudio, torchcodec, the CTranslate2 CUDA 12 libraries), the shared ffmpeg, and the shared models folder.
; It links the runtime as the runtime sub-folder of the program folder. VOXPrint MDL downloads the AI models
; as the last step into LOCALAPPDATA\Voxprint\models (one copy for every Voxprint program). shared\manifest.json
; records which programs use each resource.
; Program folder: Program Files\VOXPrint AI Suite\Movie Dubber. Launchers: voxprint-movie-dubber.exe (window)
; and voxprint-movie-dubber-cli.exe (console, same entry point as the headless CLI).
; Start menu: the folder "VOXPrint AI Suite", with a "VOXPrint AI Movie Dubber" shortcut.
; The uninstaller never touches another app's folder. A shared resource is deleted only when no program still references it.
; An older install in Program Files\VoxprintMovieDubber, and its shortcuts in the Start menu group "Voxprint",
; are removed on upgrade. User data under the Voxprint profile folder is kept. The AppId is unchanged, so Windows
; treats the new setup as an upgrade.
; Suite discovery (per-machine), until voxprint-common 0.3.0 suite.registry owns it: HKLM\Software\VOXPrint\MovieDubber
; (InstallPath, Exe, CliExe, Version, Build, IpcName, ProtocolVersion) and the two App Paths keys.
; A dev build's HKLM\Software\VOXPrint AI Suite\Apps\movie-dubber key is deleted on upgrade.
; NOTE: never write Inno constants (curly-brace names) in comments - ISCC expands some of them and the build breaks.
; Per-machine install (Program Files), Start menu entries, an entry in Apps & features and a full uninstaller.
;
; Command-line switches of the setup program:
;   /TORCH=auto|cu130         PyTorch flavor (default auto = cu130 when the GPU passes the gate)
;   /TORCH=cpu                CI only: CPU build for a GPU-less test runner. Not offered to users.
;   /SKIPGPUGATE=1            CI only, full installer only: install the cu130 runtime on a GPU-less test runner. Not offered to users.
;   /SKIPMODELS=1             CI only: do not download the AI models during setup. A user install always downloads them.
;   /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR=<folder>    unattended install

#define AppName "VoxprintMovieDubber"
#define AppDisplayName "VOXPrint AI Movie Dubber"
#define SuiteDir "VOXPrint AI Suite\Movie Dubber"
#define OldAppDir "VoxprintMovieDubber"
; APP_VERSION is read from dubber\appinfo.py. The offset and codename are read from BUILD.json.
; A CI build_info.json (build + codename) replaces the offset when the installer workflow wrote one.
#define AppInfoFile AddBackslash(SourcePath) + "..\dubber\appinfo.py"
#define BuildJsonFile AddBackslash(SourcePath) + "..\BUILD.json"
#define StampJsonFile AddBackslash(SourcePath) + "..\build_info.json"
#if FileExists(AppInfoFile) == 0
  #error dubber\appinfo.py was not found
#endif
#if FileExists(BuildJsonFile) == 0
  #error BUILD.json was not found
#endif
#define AppInfoHandle FileOpen(AppInfoFile)
#if AppInfoHandle == 0
  #error dubber\appinfo.py could not be opened
#endif
#define BuildHandle FileOpen(BuildJsonFile)
#if BuildHandle == 0
  #error BUILD.json could not be opened
#endif
; The CI stamp is opened here, before the subs that read it: ISPP needs the handle declared before a sub uses it.
#define StampHandle 0
#if FileExists(StampJsonFile) != 0
  #define StampHandle FileOpen(StampJsonFile)
#endif
#define AppVersion ""
#define AppOffset ""
#define FileCodename ""
#define StampedBuild ""
#define StampedCodename ""
#sub ParseVersionLine
  #define AppInfoLine FileRead(AppInfoHandle)
  #if AppInfoLine != ""
    #if Pos("APP_VERSION", AppInfoLine) == 1
      #define public AppVersion Copy(AppInfoLine, Pos('"', AppInfoLine) + 1, Len(AppInfoLine) - Pos('"', AppInfoLine) - 1)
    #endif
  #endif
#endsub
#sub ReadBuildJson
  #define BuildLine FileRead(BuildHandle)
  #if Pos('"offset"', BuildLine) > 0
    #define OffTail Copy(BuildLine, Pos('"offset"', BuildLine) + 8, 24)
    #define OffRaw Trim(Copy(OffTail, Pos(":", OffTail) + 1, 12))
    #define OffComma Pos(",", OffRaw)
    #if OffComma > 0
      #define OffCut Trim(Copy(OffRaw, 1, OffComma - 1))
    #else
      #define OffCut OffRaw
    #endif
    #define OffBrace Pos("}", OffCut)
    #if OffBrace > 0
      #define public AppOffset Trim(Copy(OffCut, 1, OffBrace - 1))
    #else
      #define public AppOffset OffCut
    #endif
  #endif
  #if Pos('"codename"', BuildLine) > 0
    #define NameTail Copy(BuildLine, Pos('"codename"', BuildLine) + 10, 80)
    #define NameFrom Copy(NameTail, Pos('"', NameTail) + 1, 40)
    #define public FileCodename Copy(NameFrom, 1, Pos('"', NameFrom) - 1)
  #endif
#endsub
#sub ReadStampJson
  #define StampLine FileRead(StampHandle)
  #if Pos('"build":', StampLine) > 0
    #define StampRaw Trim(Copy(StampLine, Pos('"build":', StampLine) + 8, 12))
    #define StampComma Pos(",", StampRaw)
    #if StampComma > 0
      #define StampCut Trim(Copy(StampRaw, 1, StampComma - 1))
    #else
      #define StampCut StampRaw
    #endif
    #define StampBrace Pos("}", StampCut)
    #if StampBrace > 0
      #define public StampedBuild Trim(Copy(StampCut, 1, StampBrace - 1))
    #else
      #define public StampedBuild StampCut
    #endif
  #endif
  #if Pos('"codename"', StampLine) > 0
    #define StampNameTail Copy(StampLine, Pos('"codename"', StampLine) + 10, 80)
    #define StampNameFrom Copy(StampNameTail, Pos('"', StampNameTail) + 1, 40)
    #define public StampedCodename Copy(StampNameFrom, 1, Pos('"', StampNameFrom) - 1)
  #endif
#endsub
#define AppInfoI 0
#for {AppInfoI = 0; AppInfoI < 80 && !FileEof(AppInfoHandle); AppInfoI++} ParseVersionLine
#expr FileClose(AppInfoHandle)
#define BuildI 0
#for {BuildI = 0; BuildI < 40 && !FileEof(BuildHandle); BuildI++} ReadBuildJson
#expr FileClose(BuildHandle)
#if StampHandle != 0
  #define StampI 0
  #for {StampI = 0; StampI < 40 && !FileEof(StampHandle); StampI++} ReadStampJson
  #expr FileClose(StampHandle)
#endif
#if AppVersion == ""
  #error APP_VERSION was not read from dubber\appinfo.py
#endif
#if AppOffset == ""
  #error offset was not read from BUILD.json
#endif
#if StampedBuild != ""
  #define AppBuild StampedBuild
#else
  #define AppBuild AppOffset
#endif
#if StampedCodename != ""
  #define AppCodename StampedCodename
#else
  #define AppCodename FileCodename
#endif
#define Q """
#if Pos("-rc", AppVersion) > 0
  #define Pretty Copy(AppVersion, 1, Len(AppVersion) - 3) + " RC"
  #define FileVersion Copy(AppVersion, 1, Len(AppVersion) - 3) + "." + AppBuild
#else
  #define Pretty AppVersion
  #define FileVersion AppVersion + ".0"
#endif
#define VersionLabel Pretty + " · build " + AppBuild
#if AppCodename != ""
  #define VersionLabel VersionLabel + " " + Q + AppCodename + Q
#endif
#ifdef Full
  #ifndef PayloadDir
    #define PayloadDir AddBackslash(SourcePath) + "..\build\payload"
  #endif
  #if FileExists(AddBackslash(PayloadDir) + "payload.json") == 0
    #error the full installer needs the payload (installer\make-full-payload.ps1); payload.json was not found
  #endif
  #define SetupBaseName "VoxprintMovieDubber-Full-Setup"
#else
  #define SetupBaseName "VoxprintMovieDubber-Setup"
#endif
#define PyW "{app}\runtime\Scripts\pythonw.exe"
#define Py "{app}\runtime\Scripts\python.exe"

[Setup]
AppId={{B3D84C11-52A7-4E5B-8F0D-6A9E2C41D7B5}
AppName={#AppDisplayName}
AppVersion={#AppVersion}
AppVerName={#AppDisplayName} {#VersionLabel}
VersionInfoVersion={#FileVersion}
VersionInfoProductVersion={#FileVersion}
VersionInfoProductTextVersion={#VersionLabel}
AppPublisher=VOXPrint
DefaultDirName={autopf}\{#SuiteDir}
DefaultGroupName=VOXPrint AI Suite
DisableProgramGroupPage=yes
UsePreviousAppDir=no
UsePreviousGroup=no
UninstallDisplayName={#AppDisplayName}
UninstallDisplayIcon={app}\assets\voxprint-dubber.ico
SetupIconFile=..\assets\voxprint-dubber-setup.ico
OutputDir=Output
OutputBaseFilename={#SetupBaseName}
#ifdef Full
; the payload is already compressed (wheels, tar.gz, zip): stored as is, in parts below the 2 GB GitHub asset limit
Compression=lzma2/fast
SolidCompression=no
DiskSpanning=yes
DiskSliceSize=1900000000
#else
Compression=lzma2/max
SolidCompression=yes
#endif
WizardStyle=modern
LicenseFile=..\LICENSE
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.26100
CloseApplications=yes
ExtraDiskSpaceRequired=9000000000

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
RuntimeStatus=Installing Python and PyTorch, or reusing the copy shared with other Voxprint programs (several minutes, depends on your connection)...
SiblingFound=Voxprint AI Audiobook Builder is installed on this PC: its AI models and voice library are shared with this program (nothing is downloaded twice).
RuntimeFailed=The Python environment could not be installed.%n%nCheck the internet connection and run the setup again. Details: %1
HardwareRequired=VOXPrint AI Movie Dubber needs an NVIDIA GeForce RTX 40-series graphics card or newer (Ada Lovelace or later, compute capability 8.9 or higher) and an NVIDIA driver from the 600 branch or newer.%n%nThis PC does not meet that requirement, so setup stopped and nothing was installed. A processor-only copy is not offered.%n%nInstall a supported graphics card and driver, then run setup again.%n%nNVIDIA driver downloads: https://www.nvidia.com/Download/index.aspx
UninstallDataQuestion=Also delete the dubbing projects, logs and reports (%1)?

; The two .exe launchers are compiled here, before the files are packed. csc.exe ships with the .NET Framework.
#define LaunchPs AddBackslash(SourcePath) + "launchers\write-launchers.ps1"
#define LaunchOut AddBackslash(SourcePath) + "launchers"
#define LaunchIcon AddBackslash(SourcePath) + "..\assets\voxprint-dubber.ico"
#define LaunchCmd "-NoProfile -ExecutionPolicy Bypass -File """ + LaunchPs + """ -OutDir """ + LaunchOut + """ -Icon """ + LaunchIcon + """"
#if Exec("powershell.exe", LaunchCmd, SourcePath, 1, 0) != 0
  #error the Windows launchers could not be compiled (installer\launchers\write-launchers.ps1)
#endif

[Files]
; the download script, the dependency list and the pins are only needed during the setup
Source: "install-runtime.ps1"; DestDir: "{tmp}"; Flags: dontcopy
Source: "fetch_common_wheel.py"; DestDir: "{tmp}"; Flags: dontcopy
Source: "..\requirements.txt"; DestDir: "{tmp}"; Flags: dontcopy
Source: "runtime-constraints.txt"; DestDir: "{tmp}"; Flags: dontcopy
Source: "..\dubber\infra\runtime_lock.json"; DestDir: "{tmp}"; Flags: dontcopy
Source: "..\dubber\infra\components.py"; DestDir: "{tmp}"; Flags: dontcopy
#ifdef Full
Source: "{#PayloadDir}\payload.json"; DestDir: "{tmp}\payload"; Flags: dontcopy
Source: "{#PayloadDir}\uv\*"; DestDir: "{tmp}\payload\uv"; Flags: dontcopy nocompression
Source: "{#PayloadDir}\python\*"; DestDir: "{tmp}\payload\python"; Flags: dontcopy nocompression
Source: "{#PayloadDir}\wheels\*"; DestDir: "{tmp}\payload\wheels"; Flags: dontcopy nocompression
Source: "{#PayloadDir}\ffmpeg\*"; DestDir: "{tmp}\payload\ffmpeg"; Flags: dontcopy nocompression
#endif
; the program
Source: "..\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\mdl.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "launchers\voxprint-movie-dubber.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "launchers\voxprint-movie-dubber-cli.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\BUILD.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dubber\*"; DestDir: "{app}\dubber"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\dubber\assets\icons\*.svg"; DestDir: "{app}\dubber\assets\icons"; Flags: ignoreversion
Source: "..\assets\*"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "vxdub.ico"; DestDir: "{app}\assets"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\NOTICE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\docs\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
; build stamp written by CI (commit, run number, tag); absent in a local build
Source: "..\build_info.json"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppDisplayName}"; Filename: "{app}\voxprint-movie-dubber.exe"; WorkingDir: "{app}"; IconFilename: "{app}\assets\voxprint-dubber.ico"
Name: "{group}\{cm:UninstallProgram,{#AppDisplayName}}"; Filename: "{uninstallexe}"

[Run]
; register this program as a user of the shared models folder (models\.users.json, today's store format)
Filename: "{#Py}"; Parameters: """{app}\main.py"" --register-models-user"; WorkingDir: "{app}"; StatusMsg: "Registering the shared models folder..."; Flags: runhidden runasoriginaluser
; the shared Voxprint settings (suite.json: models folder, UI language) - written only where the file has no value yet
Filename: "{#Py}"; Parameters: """{app}\main.py"" --sync-suite-settings"; WorkingDir: "{app}"; StatusMsg: "Saving the shared Voxprint settings..."; Flags: runhidden runasoriginaluser
Filename: "{app}\voxprint-movie-dubber.exe"; Parameters: "--diagnose"; WorkingDir: "{app}"; Description: "Start {#AppDisplayName} and run the diagnostics (a report is saved to the Desktop)"; Flags: nowait postinstall skipifsilent runasoriginaluser

[Registry]
; .vxdub opens in this program.  uninsdeletekey removes the type and the extension on uninstall.
Root: HKLM; Subkey: "Software\Classes\.vxdub"; ValueType: string; ValueName: ""; ValueData: "Voxprint.MovieDubber.Project"; Flags: uninsdeletekey
Root: HKLM; Subkey: "Software\Classes\.vxdub"; ValueType: string; ValueName: "Content Type"; ValueData: "application/vnd.voxprint.dub+zip"
Root: HKLM; Subkey: "Software\Classes\Voxprint.MovieDubber.Project"; ValueType: string; ValueName: ""; ValueData: "VOXPrint dubbing project"; Flags: uninsdeletekey
Root: HKLM; Subkey: "Software\Classes\Voxprint.MovieDubber.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\assets\vxdub.ico"
Root: HKLM; Subkey: "Software\Classes\Voxprint.MovieDubber.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\voxprint-movie-dubber.exe"" ""%1"""
; TODO: voxprint-common 0.3.0 suite.registry will own this key. Setup writes it until that helper exists.
; Per-machine discovery. Exe and CliExe are full paths. IpcName is empty and ProtocolVersion is 0 (no IPC server yet).
; Build is the number from BUILD.json, or the CI stamp when one was present at compile time.
; uninsdeletekey removes MovieDubber. The parent Software\VOXPrint key is removed afterwards only when it is empty.
Root: HKLM64; Subkey: "Software\VOXPrint\MovieDubber"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: uninsdeletekey
Root: HKLM64; Subkey: "Software\VOXPrint\MovieDubber"; ValueType: string; ValueName: "Exe"; ValueData: "{app}\voxprint-movie-dubber.exe"
Root: HKLM64; Subkey: "Software\VOXPrint\MovieDubber"; ValueType: string; ValueName: "CliExe"; ValueData: "{app}\voxprint-movie-dubber-cli.exe"
Root: HKLM64; Subkey: "Software\VOXPrint\MovieDubber"; ValueType: string; ValueName: "Version"; ValueData: "{#AppVersion}"
Root: HKLM64; Subkey: "Software\VOXPrint\MovieDubber"; ValueType: string; ValueName: "Build"; ValueData: "{#AppBuild}"
Root: HKLM64; Subkey: "Software\VOXPrint\MovieDubber"; ValueType: string; ValueName: "IpcName"; ValueData: ""
Root: HKLM64; Subkey: "Software\VOXPrint\MovieDubber"; ValueType: dword; ValueName: "ProtocolVersion"; ValueData: "0"
; App Paths: the default value is the full path, Path is the install directory. uninsdeletekey removes each key.
Root: HKLM64; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\voxprint-movie-dubber.exe"; ValueType: string; ValueName: ""; ValueData: "{app}\voxprint-movie-dubber.exe"; Flags: uninsdeletekey
Root: HKLM64; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\voxprint-movie-dubber.exe"; ValueType: string; ValueName: "Path"; ValueData: "{app}"
Root: HKLM64; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\voxprint-movie-dubber-cli.exe"; ValueType: string; ValueName: ""; ValueData: "{app}\voxprint-movie-dubber-cli.exe"; Flags: uninsdeletekey
Root: HKLM64; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\voxprint-movie-dubber-cli.exe"; ValueType: string; ValueName: "Path"; ValueData: "{app}"

[InstallDelete]
; Start menu folder used before 0.1.0-pre.3, and this app's shortcuts in the old "Voxprint" group.
; The program folder itself is removed in MigrateOldInstall, after the runtime junction is unlinked.
Type: filesandordirs; Name: "{commonprograms}\Voxprint AI Movie Dubber"
Type: files; Name: "{commonprograms}\Voxprint\Voxprint AI Movie Dubber.lnk"
Type: files; Name: "{commonprograms}\Voxprint\Voxprint AI Movie Dubber - diagnostics.lnk"
Type: files; Name: "{commonprograms}\Voxprint\Uninstall Voxprint AI Movie Dubber.lnk"
Type: files; Name: "{group}\Run diagnostics.lnk"
Type: files; Name: "{group}\{#AppDisplayName} - diagnostics.lnk"

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
function CmdParam(const Name, Default: String): String;
var
  I: Integer;
  Prefix: String;
begin
  Result := Default;
  Prefix := '/' + Uppercase(Name) + '=';
  for I := 1 to ParamCount do
    if Pos(Prefix, Uppercase(ParamStr(I))) = 1 then
    begin
      Result := Copy(ParamStr(I), Length(Prefix) + 1, MaxInt);
      Exit;
    end;
end;

{ Runs before the program files are copied: fetches Python, PyTorch and the dependencies.  A non-empty result stops the setup. }
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Script, Req, LogFile, Params, HwFile, Hw: String;
  Rc: Integer;
  Page: TOutputMarqueeProgressWizardPage;
  Show: Integer;
  Raw: AnsiString;
begin
  Result := '';
  ExtractTemporaryFile('install-runtime.ps1');
  ExtractTemporaryFile('fetch_common_wheel.py');
  ExtractTemporaryFile('requirements.txt');
  ExtractTemporaryFile('runtime-constraints.txt');
  ExtractTemporaryFile('runtime_lock.json');
  Script := ExpandConstant('{tmp}\install-runtime.ps1');
  Req := ExpandConstant('{tmp}\requirements.txt');
  LogFile := ExpandConstant('{%TEMP}\VoxprintMovieDubber-setup.log');
  HwFile := ExpandConstant('{%TEMP}\VoxprintMovieDubber-hardware.txt');
  ExtractTemporaryFile('components.py');
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' + Script + '" -AppDir "' + ExpandConstant('{app}') +
            '" -Requirements "' + Req + '" -Constraints "' + ExpandConstant('{tmp}\runtime-constraints.txt') +
            '" -Lock "' + ExpandConstant('{tmp}\runtime_lock.json') +
            '" -ManifestScript "' + ExpandConstant('{tmp}\components.py') +
            '" -Backend "' + CmdParam('TORCH', 'auto') + '" -HardwareFile "' + HwFile + '" -Log "' + LogFile + '"';
  if CmdParam('SKIPMODELS', '') = '1' then
    Params := Params + ' -SkipModels';
#ifdef Full
  Params := Params + ' -Payload "' + ExpandConstant('{tmp}\payload') + '"';
  if CmdParam('SKIPGPUGATE', '') = '1' then
    Params := Params + ' -SkipGpuGate';
#endif
  if WizardSilent then Show := SW_HIDE else Show := SW_SHOWNORMAL;   { the console shows the download progress }
  Page := CreateOutputMarqueeProgressPage('Installing', CustomMessage('RuntimeStatus'));
  Page.Show;
  try
    Page.Animate;
#ifdef Full
    { the payload (several GB) is unpacked from the setup parts first }
    ExtractTemporaryFiles('{tmp}\payload\payload.json');
    ExtractTemporaryFiles('{tmp}\payload\uv\*');
    ExtractTemporaryFiles('{tmp}\payload\python\*');
    ExtractTemporaryFiles('{tmp}\payload\wheels\*');
    ExtractTemporaryFiles('{tmp}\payload\ffmpeg\*');
#endif
    if not Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Params, '', Show, ewWaitUntilTerminated, Rc) then
      Rc := -1;
  finally
    Page.Hide;
  end;
  if Rc = 2 then
  begin
    Result := CustomMessage('HardwareRequired');
    Hw := '';
    if LoadStringFromFile(HwFile, Raw) then
      Hw := Trim(String(Raw));
    if Hw <> '' then
      Result := Hw;
  end
  else if Rc <> 0 then
    Result := FmtMessage(CustomMessage('RuntimeFailed'), [LogFile]);
end;

function ShouldFetchModels: Boolean;
begin
  Result := CmdParam('SKIPMODELS', '') <> '1';
end;

{ Voxprint AI Audiobook Builder: its Apps and features entry, or its name in shared\manifest.json. }
function SiblingInstalled: Boolean;
var
  Key, Manifest: String;
  Raw: AnsiString;
begin
  Key := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{6F1D2B7A-3C54-4E0B-9A41-7B5E0C9D2F18}_is1';
  Result := RegKeyExists(HKLM64, Key) or RegKeyExists(HKLM32, Key) or RegKeyExists(HKCU, Key);
  if not Result then
  begin
    Manifest := ExpandConstant('{localappdata}\Voxprint\shared\manifest.json');
    if FileExists(Manifest) and LoadStringFromFile(Manifest, Raw) then
      Result := Pos('audiobook-builder', String(Raw)) > 0;
  end;
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo, MemoComponentsInfo,
  MemoGroupInfo, MemoTasksInfo: String): String;
begin
  Result := MemoDirInfo + NewLine + NewLine + MemoGroupInfo;
  if MemoTasksInfo <> '' then
    Result := Result + NewLine + NewLine + MemoTasksInfo;
  if SiblingInstalled then
    Result := Result + NewLine + NewLine + CustomMessage('SiblingFound');
end;

{ Drop this program's references. Directories no app still uses are deleted here, except the runtime this
  process is running from: its path is written to PendingFile and removed after this process exits.
  The junction is only a link. SuppressibleMsgBox: a silent uninstall keeps the user's projects (default No). }
var
  PendingFile: String;

procedure ReleaseShared;
var
  Rc: Integer;
begin
  PendingFile := ExpandConstant('{tmp}\vmd-shared-pending.txt');
  Rc := -1;
  if FileExists(ExpandConstant('{#Py}')) then
    if not Exec(ExpandConstant('{#Py}'), '"' + ExpandConstant('{app}\main.py') + '" --release-shared --out "' + PendingFile + '"',
            ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Rc) then
      Rc := -1;
  Log(Format('shared resources: release rc=%d, pending file=%s', [Rc, PendingFile]));
  { runtime-dir.txt in the program folder names the shared runtime; the junction is only a link }
  RemoveDir(ExpandConstant('{app}\runtime'));
end;

{ The models task runs voxprint-mdl (enqueue, then status --wait) unless SKIPMODELS is 1.
  A Hugging Face token is cleared in the command file. The app does not download. }
function SameDir(const A, B: String): Boolean;
begin
  Result := CompareText(RemoveBackslash(A), RemoveBackslash(B)) = 0;
end;

function IsUserData(const Dir: String): Boolean;
var
  Root: String;
begin
  Root := Lowercase(AddBackslash(ExpandConstant('{localappdata}')));
  Result := (Dir <> '') and (Pos(Root, Lowercase(AddBackslash(Dir))) = 1);
end;

{ Drop the runtime junction first so the shared runtime under the user profile is not deleted. }
procedure RemoveProgramDir(const Dir: String);
begin
  if (Dir = '') or (not DirExists(Dir)) then Exit;
  if IsUserData(Dir) then
  begin
    Log('keeping user data at ' + Dir);
    Exit;
  end;
  RemoveDir(AddBackslash(Dir) + 'runtime');
  if not DelTree(Dir, True, True, True) then
    Log('could not remove the previous program folder ' + Dir)
  else
    Log('removed the previous program folder ' + Dir);
end;

procedure DeleteRegistryKey(const Root: Integer; const Key: String);
begin
  if RegKeyExists(Root, Key) then
    RegDeleteKeyIncludingSubkeys(Root, Key);
end;

function KeyIsEmpty(const Root: Integer; const Key: String): Boolean;
var
  Subkeys, Values: TArrayOfString;
begin
  Result := False;
  if not RegGetSubkeyNames(Root, Key, Subkeys) then Exit;
  if GetArrayLength(Subkeys) > 0 then Exit;
  if not RegGetValueNames(Root, Key, Values) then Exit;
  Result := GetArrayLength(Values) = 0;
end;

procedure DeleteKeyIfEmpty(const Root: Integer; const Key: String);
begin
  if RegKeyExists(Root, Key) and KeyIsEmpty(Root, Key) then
    RegDeleteKeyIncludingSubkeys(Root, Key);
end;

{ Dev builds wrote Software\VOXPrint AI Suite\Apps\movie-dubber. Drop that key, then empty parents. }
procedure RemoveOldSuiteKey;
begin
  DeleteRegistryKey(HKLM64, 'Software\VOXPrint AI Suite\Apps\movie-dubber');
  DeleteRegistryKey(HKLM, 'Software\VOXPrint AI Suite\Apps\movie-dubber');
  DeleteKeyIfEmpty(HKLM64, 'Software\VOXPrint AI Suite\Apps');
  DeleteKeyIfEmpty(HKLM, 'Software\VOXPrint AI Suite\Apps');
  DeleteKeyIfEmpty(HKLM64, 'Software\VOXPrint AI Suite');
  DeleteKeyIfEmpty(HKLM, 'Software\VOXPrint AI Suite');
end;

{ Leave Software\VOXPrint when another app still has a subkey or a value. }
procedure RemoveEmptyVoxprintParent;
begin
  DeleteKeyIfEmpty(HKLM64, 'Software\VOXPrint');
  DeleteKeyIfEmpty(HKLM, 'Software\VOXPrint');
end;

procedure MigrateOldInstall;
var
  OldDir, Previous, NewDir, Group, Key: String;
begin
  NewDir := ExpandConstant('{app}');
  OldDir := ExpandConstant('{autopf}\{#OldAppDir}');
  if not SameDir(OldDir, NewDir) then
    RemoveProgramDir(OldDir);
  Previous := '';
  Key := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{B3D84C11-52A7-4E5B-8F0D-6A9E2C41D7B5}_is1';
  if not RegQueryStringValue(HKLM64, Key, 'InstallLocation', Previous) then
    RegQueryStringValue(HKLM, Key, 'InstallLocation', Previous);
  if (Previous <> '') and (not SameDir(Previous, NewDir)) and (not SameDir(Previous, OldDir)) then
    RemoveProgramDir(Previous);
  Group := ExpandConstant('{commonprograms}\Voxprint');
  DeleteFile(Group + '\Voxprint AI Movie Dubber.lnk');
  DeleteFile(Group + '\Voxprint AI Movie Dubber - diagnostics.lnk');
  DeleteFile(Group + '\Uninstall Voxprint AI Movie Dubber.lnk');
  RemoveDir(Group);
  RemoveOldSuiteKey;
  Log('User data in the Voxprint profile folder is kept');
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Rc: Integer;
  Report, Params, Body: String;
  Raw: AnsiString;
  Page: TOutputProgressWizardPage;
begin
  if CurStep = ssInstall then
  begin
    MigrateOldInstall;
    Exit;
  end;
  if CurStep <> ssPostInstall then
    Exit;
  { The [Registry] entry also writes IpcName. Repeat it here so an empty string is stored. }
  RegWriteStringValue(HKLM64, 'Software\VOXPrint\MovieDubber', 'IpcName', '');
  if not ShouldFetchModels then
    Exit;
  Report := ExpandConstant('{%TEMP}\VoxprintMovieDubber-models.txt');
  Params := ExpandConstant('{%TEMP}\VoxprintMovieDubber-mdl.cmd');
  SaveStringToFile(Params,
    '@echo off' + #13#10 +
    'set HF_TOKEN=' + #13#10 +
    'set HUGGINGFACE_HUB_TOKEN=' + #13#10 +
    '"' + ExpandConstant('{#Py}') + '" -m voxprint_common.mdl enqueue --app movie-dubber --manifest "' + ExpandConstant('{app}\mdl.json') + '" > "' + Report + '" 2>&1' + #13#10 +
    'if errorlevel 1 exit /b %errorlevel%' + #13#10 +
    '"' + ExpandConstant('{#Py}') + '" -m voxprint_common.mdl --json status --wait --manifest "' + ExpandConstant('{app}\mdl.json') + '" >> "' + Report + '" 2>&1' + #13#10 +
    'exit /b %errorlevel%' + #13#10, False);
  Page := CreateOutputProgressPage('AI models', 'VOXPrint MDL is downloading the models for this computer...');
  Page.Show;
  try
    if not Exec(Params, '', ExpandConstant('{app}'), SW_SHOWNORMAL, ewWaitUntilTerminated, Rc) then
      Rc := -1;
  finally
    Page.Hide;
  end;
  Body := '';
  if LoadStringFromFile(Report, Raw) then
    Body := Trim(String(Raw));
  if Rc <> 0 then
  begin
    if Body = '' then
      Body := 'VOXPrint MDL failed and wrote no report. Run the setup again, or use Manage models in the program.';
    RaiseException(Body);
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir, Line, Root: String;
  Lines: TArrayOfString;
  I: Integer;
begin
  if CurUninstallStep = usUninstall then
    ReleaseShared;
  if CurUninstallStep = usPostUninstall then
  begin
    RemoveOldSuiteKey;
    RemoveEmptyVoxprintParent;
    DataDir := ExpandConstant('{localappdata}\{#AppName}');
    if DirExists(DataDir) then
      if SuppressibleMsgBox(FmtMessage(CustomMessage('UninstallDataQuestion'), [DataDir]),
         mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
        DelTree(DataDir, True, True, True);
    Root := Lowercase(ExpandConstant('{localappdata}\Voxprint\shared\'));
    if (PendingFile <> '') and LoadStringsFromFile(PendingFile, Lines) then
      for I := 0 to GetArrayLength(Lines) - 1 do
      begin
        Line := Trim(Lines[I]);
        if (Line <> '') and (Pos(Root, Lowercase(Line)) = 1) and DirExists(Line) then
          if not DelTree(Line, True, True, True) then Log('shared: could not delete ' + Line);
      end;
  end;
end;
