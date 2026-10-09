; Voxprint AI Movie Dubber - ONLINE installer (Inno Setup 6.x).
;   ISCC installer\VoxprintMovieDubber.iss   ->   installer\Output\VoxprintMovieDubber-Setup.exe
;
; The installer is small: it contains only the program's own files (Python sources, icon, licences) and a download script.
; While it installs, install-runtime.ps1 fetches a private Python 3.11, PyTorch (the build that matches the NVIDIA driver) and the
; other dependencies from the internet into the runtime sub-folder of the program folder; the AI models are downloaded as the
; last (optional) step into the models folder shared with Voxprint AI Audiobook Builder (one copy for both programs).
; NOTE: never write Inno constants (curly-brace names) in comments - ISCC expands some of them and the build breaks.
; Per-machine install (Program Files), Start menu entries, an entry in Apps & features and a full uninstaller.
;
; Command-line switches of the setup program:
;   /TORCH=auto|cpu|cu128    PyTorch build (default: auto = matches the driver, CPU when there is no NVIDIA GPU)
;   /TASKS=""                do not download the AI models during the setup (they are downloaded when first needed)
;   /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR=<folder>    unattended install

#define AppName "VoxprintMovieDubber"
#define AppDisplayName "Voxprint AI Movie Dubber"
#define AppVersion "0.1.0"
#define PyW "{app}\runtime\Scripts\pythonw.exe"
#define Py "{app}\runtime\Scripts\python.exe"

[Setup]
AppId={{B3D84C11-52A7-4E5B-8F0D-6A9E2C41D7B5}
AppName={#AppDisplayName}
AppVersion={#AppVersion}
AppVerName={#AppDisplayName} {#AppVersion}
AppPublisher=Voxprint
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppDisplayName}
DisableProgramGroupPage=yes
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
MinVersion=10.0.19041
CloseApplications=yes
ExtraDiskSpaceRequired=9000000000

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
RuntimeStatus=Downloading and installing Python and PyTorch (several minutes, depends on your connection)...
RuntimeFailed=The Python environment could not be installed.%n%nCheck the internet connection and run the setup again. Details: %1
UninstallDataQuestion=Also delete the dubbing projects, logs and reports (%1)?
UninstallModelsQuestion=No other Voxprint program uses the downloaded AI models any more.%n%nDelete them too (%1, several gigabytes)? Choose No to keep them for a later installation.

[Tasks]
Name: "models"; Description: "Download the AI models now (about 8 GB, one time; otherwise they are downloaded when first needed)"; GroupDescription: "AI models:"

[Files]
; the download script and the dependency list are only needed during the setup
Source: "install-runtime.ps1"; DestDir: "{tmp}"; Flags: dontcopy
Source: "..\requirements.txt"; DestDir: "{tmp}"; Flags: dontcopy
; the program
Source: "..\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dubber\*"; DestDir: "{app}\dubber"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\assets\*"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\docs\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
; build stamp written by CI (commit, run number, tag); absent in a local build
Source: "..\build_info.json"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppDisplayName}"; Filename: "{#PyW}"; Parameters: """{app}\main.py"""; WorkingDir: "{app}"; IconFilename: "{app}\assets\voxprint-dubber.ico"
Name: "{group}\Run diagnostics"; Filename: "{#PyW}"; Parameters: """{app}\main.py"" --diagnose"; WorkingDir: "{app}"; IconFilename: "{app}\assets\voxprint-dubber.ico"
Name: "{group}\{cm:UninstallProgram,{#AppDisplayName}}"; Filename: "{uninstallexe}"

[Run]
; register this program as a user of the shared models folder (models\.users.json)
Filename: "{#Py}"; Parameters: """{app}\main.py"" --register-models-user"; WorkingDir: "{app}"; StatusMsg: "Registering the shared models folder..."; Flags: runhidden runasoriginaluser
Filename: "{#Py}"; Parameters: """{app}\main.py"" --fetch-models"; WorkingDir: "{app}"; Tasks: models; StatusMsg: "Downloading the AI models (this can take a while)..."; Flags: runasoriginaluser
Filename: "{#PyW}"; Parameters: """{app}\main.py"" --diagnose"; WorkingDir: "{app}"; Description: "Start {#AppDisplayName} and run the diagnostics (a report is saved to the Desktop)"; Flags: nowait postinstall skipifsilent runasoriginaluser

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
  Script, Req, LogFile, Params: String;
  Rc: Integer;
  Page: TOutputMarqueeProgressWizardPage;
  Show: Integer;
begin
  Result := '';
  ExtractTemporaryFile('install-runtime.ps1');
  ExtractTemporaryFile('requirements.txt');
  Script := ExpandConstant('{tmp}\install-runtime.ps1');
  Req := ExpandConstant('{tmp}\requirements.txt');
  LogFile := ExpandConstant('{%TEMP}\VoxprintMovieDubber-setup.log');
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' + Script + '" -AppDir "' + ExpandConstant('{app}') +
            '" -Requirements "' + Req + '" -Backend "' + CmdParam('TORCH', 'auto') + '" -Log "' + LogFile + '"';
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
  if Rc <> 0 then
    Result := FmtMessage(CustomMessage('RuntimeFailed'), [LogFile]);
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

{ User data is only removed when the user agrees; the shared models only when no other Voxprint program uses them (default: keep). }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usUninstall then
    UnregisterModelsUser;
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\{#AppName}');
    if DirExists(DataDir) then
      if MsgBox(FmtMessage(CustomMessage('UninstallDataQuestion'), [DataDir]),
         mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(DataDir, True, True, True);
    if (OtherUsers = 0) and (ModelsDir <> '') and DirExists(ModelsDir) then
      if MsgBox(FmtMessage(CustomMessage('UninstallModelsQuestion'), [ModelsDir]),
         mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(ModelsDir, True, True, True);
  end;
end;
