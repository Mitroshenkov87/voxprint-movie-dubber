; Voxprint AI Movie Dubber - ONLINE installer (Inno Setup 6.x).
;   ISCC installer\VoxprintMovieDubber.iss   ->   installer\Output\VoxprintMovieDubber-Setup.exe
;
; The installer is small: it contains only the program's own files (Python sources, icon, licences) and a download script.
; While it installs, install-runtime.ps1 installs (or reuses, on an exact match of the pinned versions) the shared runtime
; LOCALAPPDATA\Voxprint\runtime-<key> (Python 3.14, PyTorch 2.11.0+cu130, the other
; dependencies) and links it as the runtime sub-folder of the program folder; the AI models are downloaded as the last
; (optional) step into the models folder shared with Voxprint AI Audiobook Builder (one copy for both programs).
; Start menu: the folder "Voxprint", shared with the Audiobook Builder.  The uninstaller never touches the Audiobook Builder's
; files; shared models are deleted only when no other program uses them and the user agrees (default: keep).
; NOTE: never write Inno constants (curly-brace names) in comments - ISCC expands some of them and the build breaks.
; Per-machine install (Program Files), Start menu entries, an entry in Apps & features and a full uninstaller.
;
; Command-line switches of the setup program:
;   /TORCH=auto|cu130         PyTorch flavor (default auto = cu130 when the GPU passes the gate)
;   /TORCH=cpu                CI only: CPU build for a GPU-less test runner. Not offered to users.
;   /TASKS=""                do not download the AI models during the setup (they are downloaded when first needed)
;   /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR=<folder>    unattended install

#define AppName "VoxprintMovieDubber"
#define AppDisplayName "Voxprint AI Movie Dubber"
; APP_VERSION, APP_BUILD and CODENAME are read from dubber\appinfo.py (the only copy).
#define AppInfoFile AddBackslash(SourcePath) + "..\dubber\appinfo.py"
#if FileExists(AppInfoFile) == 0
  #error dubber\appinfo.py was not found
#endif
#define AppInfoHandle FileOpen(AppInfoFile)
#if AppInfoHandle == 0
  #error dubber\appinfo.py could not be opened
#endif
#define AppVersion ""
#define AppBuild ""
#define AppCodename ""
#sub ParseAppInfoLine
  #define AppInfoLine FileRead(AppInfoHandle)
  #if AppInfoLine != ""
    #if Pos("APP_VERSION", AppInfoLine) == 1
      #define public AppVersion Copy(AppInfoLine, Pos('"', AppInfoLine) + 1, Len(AppInfoLine) - Pos('"', AppInfoLine) - 1)
    #elif Pos("APP_BUILD", AppInfoLine) == 1
      #define public AppBuild Trim(Copy(AppInfoLine, Pos("=", AppInfoLine) + 1, 16))
    #elif Pos("CODENAME", AppInfoLine) == 1
      #define public AppCodename Copy(AppInfoLine, Pos('"', AppInfoLine) + 1, Len(AppInfoLine) - Pos('"', AppInfoLine) - 1)
    #endif
  #endif
#endsub
#define AppInfoI 0
#for {AppInfoI = 0; AppInfoI < 80 && !FileEof(AppInfoHandle); AppInfoI++} ParseAppInfoLine
#expr FileClose(AppInfoHandle)
#if AppVersion == ""
  #error APP_VERSION was not read from dubber\appinfo.py
#endif
#if AppBuild == ""
  #error APP_BUILD was not read from dubber\appinfo.py
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
OutputBaseFilename=VoxprintMovieDubber-Setup
Compression=lzma2/max
SolidCompression=yes
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
HardwareRequired=Voxprint AI Movie Dubber needs an NVIDIA GeForce RTX 40-series graphics card or newer (Ada Lovelace or later, compute capability 8.9 or higher) and an NVIDIA driver from the 600 branch or newer.%n%nThis PC does not meet that requirement, so setup stopped and nothing was installed. A processor-only copy is not offered.%n%nInstall a supported graphics card and driver, then run setup again.
UninstallDataQuestion=Also delete the dubbing projects, logs and reports (%1)?
UninstallModelsQuestion=No other Voxprint program uses the downloaded AI models any more.%n%nDelete them too (%1, several gigabytes)? Choose No to keep them for a later installation.

[Tasks]
Name: "models"; Description: "Download the AI models now (about 8 GB, one time; otherwise they are downloaded when first needed)"; GroupDescription: "AI models:"

[Files]
; the download script, the dependency list and the pins are only needed during the setup
Source: "install-runtime.ps1"; DestDir: "{tmp}"; Flags: dontcopy
Source: "..\requirements.txt"; DestDir: "{tmp}"; Flags: dontcopy
Source: "runtime-constraints.txt"; DestDir: "{tmp}"; Flags: dontcopy
Source: "..\dubber\infra\runtime_lock.json"; DestDir: "{tmp}"; Flags: dontcopy
; the program
Source: "..\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dubber\*"; DestDir: "{app}\dubber"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\assets\*"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs createallsubdirs
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
; register this program as a user of the shared models folder (models\.users.json)
Filename: "{#Py}"; Parameters: """{app}\main.py"" --register-models-user"; WorkingDir: "{app}"; StatusMsg: "Registering the shared models folder..."; Flags: runhidden runasoriginaluser
; the shared Voxprint settings (suite.json: models folder, UI language) - written only where the file has no value yet
Filename: "{#Py}"; Parameters: """{app}\main.py"" --sync-suite-settings"; WorkingDir: "{app}"; StatusMsg: "Saving the shared Voxprint settings..."; Flags: runhidden runasoriginaluser
Filename: "{#Py}"; Parameters: """{app}\main.py"" --fetch-models"; WorkingDir: "{app}"; Tasks: models; StatusMsg: "Downloading the AI models (this can take a while)..."; Flags: runasoriginaluser
Filename: "{#PyW}"; Parameters: """{app}\main.py"" --diagnose"; WorkingDir: "{app}"; Description: "Start {#AppDisplayName} and run the diagnostics (a report is saved to the Desktop)"; Flags: nowait postinstall skipifsilent runasoriginaluser

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
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' + Script + '" -AppDir "' + ExpandConstant('{app}') +
            '" -Requirements "' + Req + '" -Constraints "' + ExpandConstant('{tmp}\runtime-constraints.txt') +
            '" -Lock "' + ExpandConstant('{tmp}\runtime_lock.json') +
            '" -Backend "' + CmdParam('TORCH', 'auto') + '" -HardwareFile "' + HwFile + '" -Log "' + LogFile + '"';
  if WizardSilent then Show := SW_HIDE else Show := SW_SHOWNORMAL;   { the console shows the download progress }
  Page := CreateOutputMarqueeProgressPage('Installing', CustomMessage('RuntimeStatus'));
  Page.Show;
  try
    Page.Animate;
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

{ Voxprint AI Audiobook Builder (the sibling program): its Apps and features entry or its key in the shared models\.users.json. }
function SiblingInstalled: Boolean;
var
  Key, Users: String;
  Raw: AnsiString;
begin
  Key := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{6F1D2B7A-3C54-4E0B-9A41-7B5E0C9D2F18}_is1';
  Result := RegKeyExists(HKLM64, Key) or RegKeyExists(HKLM32, Key) or RegKeyExists(HKCU, Key);
  if not Result then
  begin
    Users := ExpandConstant('{localappdata}\Voxprint\models\.users.json');
    if FileExists(Users) and LoadStringFromFile(Users, Raw) then
      Result := Pos('"audiobook-builder": true', String(Raw)) > 0;
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

{ Before the files go: remove our key from the shared models\.users.json.  The helper writes two lines to a temp file:
  the number of other programs still using the models and the models folder. }
var
  OtherUsers: Integer;
  ModelsDir: String;

procedure UnregisterModelsUser;
var
  Rc: Integer;
  OutFile: String;
  Lines: TArrayOfString;
begin
  OtherUsers := -1;
  ModelsDir := '';
  OutFile := ExpandConstant('{tmp}\vmd-models-users.txt');
  if not FileExists(ExpandConstant('{#Py}')) then Exit;
  if Exec(ExpandConstant('{#Py}'), '"' + ExpandConstant('{app}\main.py') + '" --unregister-models-user --out "' + OutFile + '"',
          ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Rc) and (Rc = 0) then
    if LoadStringsFromFile(OutFile, Lines) and (GetArrayLength(Lines) >= 2) then
    begin
      OtherUsers := StrToIntDef(Lines[0], -1);
      ModelsDir := Lines[1];
    end;
end;

{ User data is only removed when the user agrees; the shared models only when no other Voxprint program uses them (default: keep).
  SuppressibleMsgBox: a silent uninstall (/SUPPRESSMSGBOXES) takes the default answer - keep. }
var
  RuntimeOthers: Integer;
  RuntimeDir: String;

{ Our key out of runtime-<key>\.users.json (the runtime is shared by key); then the junction <app>\runtime is removed (only the link,
  never the shared folder it points to). }
procedure UnregisterRuntimeUser;
var
  Rc: Integer;
  OutFile: String;
  Lines: TArrayOfString;
begin
  Rc := -1;
  RuntimeOthers := -1;
  RuntimeDir := '';
  OutFile := ExpandConstant('{tmp}\vmd-runtime-users.txt');
  if FileExists(ExpandConstant('{#Py}')) then
    if Exec(ExpandConstant('{#Py}'), '"' + ExpandConstant('{app}\main.py') + '" --unregister-runtime-user --out "' + OutFile + '"',
            ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Rc) and (Rc = 0) then
      if LoadStringsFromFile(OutFile, Lines) and (GetArrayLength(Lines) >= 2) then
      begin
        RuntimeOthers := StrToIntDef(Trim(Lines[0]), -1);
        RuntimeDir := Trim(Lines[1]);
      end;
  { the installer's own note of the real runtime-<key> folder, in case the interpreter could not tell }
  if (RuntimeDir = '') and LoadStringsFromFile(ExpandConstant('{app}\runtime-dir.txt'), Lines) and (GetArrayLength(Lines) >= 1) then
    RuntimeDir := Trim(Lines[0]);
  Log(Format('runtime: dir=%s, other users=%d (python rc=%d)', [RuntimeDir, RuntimeOthers, Rc]));
  RemoveDir(ExpandConstant('{app}\runtime'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usUninstall then
  begin
    UnregisterModelsUser;
    UnregisterRuntimeUser;
  end;
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\{#AppName}');
    if DirExists(DataDir) then
      if SuppressibleMsgBox(FmtMessage(CustomMessage('UninstallDataQuestion'), [DataDir]),
         mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
        DelTree(DataDir, True, True, True);
    { the shared runtime goes only when no other program is listed in its .users.json (it is never the Audiobook Builder's own
      runtime folder: ours are named runtime-<key>) }
    if (RuntimeOthers = 0) and (RuntimeDir <> '') and DirExists(RuntimeDir) and
       (Pos('\runtime-', RuntimeDir) > 0) and FileExists(AddBackslash(RuntimeDir) + 'runtime-key.json') then
    begin
      if not DelTree(RuntimeDir, True, True, True) then Log('runtime: could not delete ' + RuntimeDir);
    end;
    if (OtherUsers = 0) and (ModelsDir <> '') and DirExists(ModelsDir) then
      if SuppressibleMsgBox(FmtMessage(CustomMessage('UninstallModelsQuestion'), [ModelsDir]),
         mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
        DelTree(ModelsDir, True, True, True);
  end;
end;
