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
; It links the runtime as the runtime sub-folder of the program folder. The AI models are downloaded as the last
; step into LOCALAPPDATA\Voxprint\shared\models (one copy for every Voxprint program). shared\manifest.json
; records which programs use each resource.
; Start menu: the folder "Voxprint", shared with the Audiobook Builder. The uninstaller never touches the Audiobook Builder's
; own folder. A shared resource is deleted only when no program still references it.
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
#define AppDisplayName "Voxprint AI Movie Dubber"
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
AppPublisher=Voxprint
DefaultDirName={autopf}\{#AppName}
DefaultGroupName=Voxprint
DisableProgramGroupPage=yes
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
HardwareRequired=Voxprint AI Movie Dubber needs an NVIDIA GeForce RTX 40-series graphics card or newer (Ada Lovelace or later, compute capability 8.9 or higher) and an NVIDIA driver from the 600 branch or newer.%n%nThis PC does not meet that requirement, so setup stopped and nothing was installed. A processor-only copy is not offered.%n%nInstall a supported graphics card and driver, then run setup again.%n%nNVIDIA driver downloads: https://www.nvidia.com/Download/index.aspx
UninstallDataQuestion=Also delete the dubbing projects, logs and reports (%1)?

[Files]
; the download script, the dependency list and the pins are only needed during the setup
Source: "install-runtime.ps1"; DestDir: "{tmp}"; Flags: dontcopy
Source: "..\requirements.txt"; DestDir: "{tmp}"; Flags: dontcopy
Source: "runtime-constraints.txt"; DestDir: "{tmp}"; Flags: dontcopy
Source: "..\dubber\infra\runtime_lock.json"; DestDir: "{tmp}"; Flags: dontcopy
Source: "..\dubber\infra\shared_manifest.py"; DestDir: "{tmp}"; Flags: dontcopy
#ifdef Full
Source: "{#PayloadDir}\payload.json"; DestDir: "{tmp}\payload"; Flags: dontcopy
Source: "{#PayloadDir}\uv\*"; DestDir: "{tmp}\payload\uv"; Flags: dontcopy nocompression
Source: "{#PayloadDir}\python\*"; DestDir: "{tmp}\payload\python"; Flags: dontcopy nocompression
Source: "{#PayloadDir}\wheels\*"; DestDir: "{tmp}\payload\wheels"; Flags: dontcopy nocompression
Source: "{#PayloadDir}\ffmpeg\*"; DestDir: "{tmp}\payload\ffmpeg"; Flags: dontcopy nocompression
#endif
; the program
Source: "..\main.py"; DestDir: "{app}"; Flags: ignoreversion
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
Name: "{group}\{#AppDisplayName}"; Filename: "{#PyW}"; Parameters: """{app}\main.py"""; WorkingDir: "{app}"; IconFilename: "{app}\assets\voxprint-dubber.ico"
Name: "{group}\{#AppDisplayName} - diagnostics"; Filename: "{#PyW}"; Parameters: """{app}\main.py"" --diagnose"; WorkingDir: "{app}"; IconFilename: "{app}\assets\voxprint-dubber.ico"
Name: "{group}\{cm:UninstallProgram,{#AppDisplayName}}"; Filename: "{uninstallexe}"

[Run]
; register this program as a user of the shared models folder (models\.users.json, today's store format)
Filename: "{#Py}"; Parameters: """{app}\main.py"" --register-models-user"; WorkingDir: "{app}"; StatusMsg: "Registering the shared models folder..."; Flags: runhidden runasoriginaluser
; the shared Voxprint settings (suite.json: models folder, UI language) - written only where the file has no value yet
Filename: "{#Py}"; Parameters: """{app}\main.py"" --sync-suite-settings"; WorkingDir: "{app}"; StatusMsg: "Saving the shared Voxprint settings..."; Flags: runhidden runasoriginaluser
Filename: "{#Py}"; Parameters: """{app}\main.py"" --fetch-models"; WorkingDir: "{app}"; StatusMsg: "Downloading the AI models (this can take a while)..."; Flags: runasoriginaluser; Check: ShouldFetchModels
Filename: "{#PyW}"; Parameters: """{app}\main.py"" --diagnose"; WorkingDir: "{app}"; Description: "Start {#AppDisplayName} and run the diagnostics (a report is saved to the Desktop)"; Flags: nowait postinstall skipifsilent runasoriginaluser

[Registry]
; .vxdub opens in this program.  uninsdeletekey removes the type and the extension on uninstall.
Root: HKLM; Subkey: "Software\Classes\.vxdub"; ValueType: string; ValueName: ""; ValueData: "Voxprint.MovieDubber.Project"; Flags: uninsdeletekey
Root: HKLM; Subkey: "Software\Classes\.vxdub"; ValueType: string; ValueName: "Content Type"; ValueData: "application/vnd.voxprint.dub+zip"
Root: HKLM; Subkey: "Software\Classes\Voxprint.MovieDubber.Project"; ValueType: string; ValueName: ""; ValueData: "Voxprint dubbing project"; Flags: uninsdeletekey
Root: HKLM; Subkey: "Software\Classes\Voxprint.MovieDubber.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\assets\vxdub.ico"
Root: HKLM; Subkey: "Software\Classes\Voxprint.MovieDubber.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{#PyW}"" ""{app}\main.py"" ""%1"""

[InstallDelete]
; shortcuts of builds before 0.1.0-pre.3 (their own Start menu folder; now the shared "Voxprint" folder is used)
Type: filesandordirs; Name: "{commonprograms}\{#AppDisplayName}"
Type: files; Name: "{group}\Run diagnostics.lnk"

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
  ExtractTemporaryFile('requirements.txt');
  ExtractTemporaryFile('runtime-constraints.txt');
  ExtractTemporaryFile('runtime_lock.json');
  Script := ExpandConstant('{tmp}\install-runtime.ps1');
  Req := ExpandConstant('{tmp}\requirements.txt');
  LogFile := ExpandConstant('{%TEMP}\VoxprintMovieDubber-setup.log');
  HwFile := ExpandConstant('{%TEMP}\VoxprintMovieDubber-hardware.txt');
  ExtractTemporaryFile('shared_manifest.py');
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' + Script + '" -AppDir "' + ExpandConstant('{app}') +
            '" -Requirements "' + Req + '" -Constraints "' + ExpandConstant('{tmp}\runtime-constraints.txt') +
            '" -Lock "' + ExpandConstant('{tmp}\runtime_lock.json') +
            '" -ManifestScript "' + ExpandConstant('{tmp}\shared_manifest.py') +
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
