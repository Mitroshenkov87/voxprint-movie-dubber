; Voxprint AI Movie Dubber installer (Inno Setup 6.x).  UNTESTED (written on Linux) - see docs\BUILDING.md.
;   ISCC installer\VoxprintMovieDubber.iss          -> installer\Output\VoxprintMovieDubber-Setup.exe
; Packs the PyInstaller folder dist\VoxprintMovieDubber (build.bat exe).  The models are NOT in the installer: they are
; downloaded on the first run (like the Audiobook Builder).  User data lives in %LOCALAPPDATA%\VoxprintMovieDubber.
; Per-user install (no admin rights needed) - nothing system-wide is touched.

#define AppName "VoxprintMovieDubber"
#define AppDisplayName "Voxprint AI Movie Dubber"
#define AppVersion "0.1.0"
#define AppExe "VoxprintMovieDubber.exe"

[Setup]
AppId={{B3D84C11-52A7-4E5B-8F0D-6A9E2C41D7B5}
AppName={#AppDisplayName}
AppVersion={#AppVersion}
AppPublisher=Voxprint
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppDisplayName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExe}
OutputDir=Output
OutputBaseFilename=VoxprintMovieDubber-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[CustomMessages]
english.RunDiag=Start %1 and run the diagnostics (report is saved to the Desktop)
russian.RunDiag=Запустить %1 и сразу выполнить диагностику (отчёт сохранится на Рабочий стол)
english.DiagIcon=Run diagnostics
russian.DiagIcon=Диагностика (отчёт на Рабочий стол)

[Files]
Source: "..\dist\VoxprintMovieDubber\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppDisplayName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:DiagIcon}"; Filename: "{app}\{#AppExe}"; Parameters: "--diagnose"
Name: "{autodesktop}\{#AppDisplayName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; Flags: unchecked

[Run]
Filename: "{app}\{#AppExe}"; Parameters: "--diagnose"; Description: "{cm:RunDiag,{#AppDisplayName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
