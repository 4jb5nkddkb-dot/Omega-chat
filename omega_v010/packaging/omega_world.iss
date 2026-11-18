; Omega small world — Inno Setup script (v0.1.2).
; Built on windows-latest by .github/workflows/build-windows.yml from the
; PyInstaller onedir output (dist\OmegaWorld). Per-user install, no admin:
; the GUI writes dialog_log.jsonl and run_kernel/ next to the exe, so the
; folder must stay writable — install to {localappdata}, not Program Files.
; Pass /DBUNDLE_MODEL=1 when the .npz was bundled; the Start Menu shortcut
; then launches with --ckpt so the first run is one click, no file picker.

#define AppVersion "0.1.2"
#ifndef BundleModel
  #define BundleModel "0"
#endif
#ifndef ModelFile
  #define ModelFile "generalist2_s5100.npz"
#endif

[Setup]
AppName=Omega 小世界
AppVersion={#AppVersion}
DefaultDirName={localappdata}\OmegaWorld
DefaultGroupName=Omega 小世界
PrivilegesRequired=lowest
OutputDir=..\..\installer-out
OutputBaseFilename=OmegaWorld-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=Omega 小世界

[Files]
Source: "..\..\dist\OmegaWorld\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
#if BundleModel == "1"
Name: "{group}\Omega 小世界"; Filename: "{app}\OmegaWorld.exe"; Parameters: "--ckpt ""{app}\{#ModelFile}"""; WorkingDir: "{app}"
Name: "{autodesktop}\Omega 小世界"; Filename: "{app}\OmegaWorld.exe"; Parameters: "--ckpt ""{app}\{#ModelFile}"""; WorkingDir: "{app}"; Tasks: desktopicon
#else
Name: "{group}\Omega 小世界"; Filename: "{app}\OmegaWorld.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Omega 小世界"; Filename: "{app}\OmegaWorld.exe"; WorkingDir: "{app}"; Tasks: desktopicon
#endif

[Tasks]
Name: "desktopicon"; Description: "创建桌面图标"; GroupDescription: "附加选项："

[Run]
Filename: "{app}\OmegaWorld.exe"; Description: "现在运行 Omega 小世界"; Flags: nowait postinstall skipifsilent
