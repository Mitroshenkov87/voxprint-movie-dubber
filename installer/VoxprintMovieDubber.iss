; Voxprint AI Movie Dubber - ONLINE installer (Inno Setup 6.x).
;   ISCC installer\VoxprintMovieDubber.iss   ->   installer\Output\VoxprintMovieDubber-Setup.exe
;
; The installer is small: it contains only the program's own files (Python sources, icon, licences) and a download script.
; While it installs, install-runtime.ps1 fetches a private Python 3.11, PyTorch (the build that matches the NVIDIA driver) and the
; other dependencies from the internet into {app}\runtime; the AI models are downloaded as the last (optional) step.
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
UninstallDataQuestion=Also delete the downloaded AI models, logs and reports (%1)? They take several gigabytes.

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

[Icons]
Name: "{group}\{#AppDisplayName}"; Filename: "{#PyW}"; Parameters: """{app}\main.py"""; WorkingDir: "{app}"; IconFilename: "{app}\assets\voxprint-dubber.ico"
Name: "{group}\Run diagnostics"; Filename: "{#PyW}"; Parameters: """{app}\main.py"" --diagnose"; WorkingDir: "{app}"; IconFilename: "{app}\assets\voxprint-dubber.ico"
Name: "{group}\{cm:UninstallProgram,{#AppDisplayName}}"; Filename: "{uninstallexe}"

[Run]
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

{ The user data (models, logs, reports) is only removed when the user agrees. }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\{#AppName}');
    if DirExists(DataDir) then
      if MsgBox(FmtMessage(CustomMessage('UninstallDataQuestion'), [DataDir]),
         mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;
