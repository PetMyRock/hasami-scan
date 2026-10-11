; Inno Setup script for Hasami Scan.  build.py runs:
;   iscc /DAppVersion=0.3.15 /DAppDir=...\dist\HasamiScan /DOutDir=...\dist /DIconFile=...\hasami.ico installer.iss
; Per-user install by default (no admin prompt): %LOCALAPPDATA%\Programs\Hasami Scan.
; Settings stay in %APPDATA%\HasamiScan and are kept on uninstall.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef AppDir
  #define AppDir "..\dist\HasamiScan"
#endif
#ifndef OutDir
  #define OutDir "..\dist"
#endif

[Setup]
AppId={{6F1E8C2A-4B7D-4E39-9A51-3C2D8F0B7E14}
AppName=Hasami Scan
AppVersion={#AppVersion}
AppVerName=Hasami Scan {#AppVersion}
AppPublisher=Hasami Scan
DefaultDirName={autopf}\Hasami Scan
DefaultGroupName=Hasami Scan
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutDir}
OutputBaseFilename=HasamiScanSetup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\HasamiScan.exe
UninstallDisplayName=Hasami Scan {#AppVersion}
CloseApplications=yes
#ifdef IconFile
SetupIconFile={#IconFile}
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#AppDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; Files of an older version that the new one no longer has.
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\twain32"
Type: filesandordirs; Name: "{app}\twain64"

[Icons]
Name: "{autoprograms}\Hasami Scan"; Filename: "{app}\HasamiScan.exe"
Name: "{autodesktop}\Hasami Scan"; Filename: "{app}\HasamiScan.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\HasamiScan.exe"; Description: "{cm:LaunchProgram,Hasami Scan}"; Flags: nowait postinstall skipifsilent
