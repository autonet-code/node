# Autonet installer for Windows
# Downloads and installs the autonet package into an isolated venv.
#
# Usage (PowerShell):
#   irm https://github.com/autonet-code/node/releases/latest/download/install.ps1 | iex
#   # or:
#   .\install.ps1                  # install from PyPI
#   .\install.ps1 .\autonet.whl   # install from a local wheel
param(
    [string]$WheelPath = ""
)

$ErrorActionPreference = "Stop"

$InstallDir = if ($env:AUTONET_INSTALL_DIR) { $env:AUTONET_INSTALL_DIR } else { "$env:USERPROFILE\.autonet" }
$VenvDir = "$InstallDir\venv"
$MinPython = "3.11"

function Info($msg) { Write-Host "=> $msg" -ForegroundColor Cyan }
function Error($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# --- Find Python >= 3.11 ---
function Find-Python {
    foreach ($candidate in @("python3", "python", "py -3")) {
        try {
            $cmd = if ($candidate -eq "py -3") { "py" } else { $candidate }
            $args = if ($candidate -eq "py -3") { @("-3", "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") } else { @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") }
            $version = & $cmd @args 2>$null
            if ($version) {
                $parts = $version.Split(".")
                $major = [int]$parts[0]
                $minor = [int]$parts[1]
                if ($major -ge 3 -and $minor -ge 11) {
                    if ($candidate -eq "py -3") { return "py", "-3" }
                    return $cmd, $null
                }
            }
        } catch { }
    }
    Error "Python >= $MinPython is required but not found. Install it from python.org."
}

$pythonResult = Find-Python
$PythonCmd = $pythonResult[0]
$PythonArg = $pythonResult[1]

$versionStr = if ($PythonArg) { & $PythonCmd $PythonArg --version } else { & $PythonCmd --version }
Info "Using $PythonCmd ($versionStr)"

# --- Create venv ---
Info "Creating virtual environment at $VenvDir"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
if ($PythonArg) {
    & $PythonCmd $PythonArg -m venv $VenvDir
} else {
    & $PythonCmd -m venv $VenvDir
}

$PipExe = "$VenvDir\Scripts\pip.exe"
$AtnExe = "$VenvDir\Scripts\atn.exe"

# --- Install ---
if ($WheelPath -and (Test-Path $WheelPath)) {
    Info "Installing from local file: $WheelPath"
    & $PipExe install --upgrade $WheelPath
} else {
    Info "Installing autonet from PyPI"
    & $PipExe install --upgrade autonet
}

# --- Add to PATH ---
$VenvBin = "$VenvDir\Scripts"
$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($UserPath -notlike "*$VenvBin*") {
    [Environment]::SetEnvironmentVariable("PATH", "$VenvBin;$UserPath", "User")
    Info "Added $VenvBin to your user PATH"
    Info "Restart your terminal for PATH changes to take effect"
} else {
    Info "$VenvBin is already in PATH"
}

Info "Installation complete!"
Info "Run 'atn' to start the agent framework."
Info "Optional extras: pip install autonet[voice] autonet[node] autonet[p2p]"
