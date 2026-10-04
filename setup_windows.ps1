# ==============================================================================
# Desktop VLM Lens - 1-Click Windows Setup & Packaging Wizard
# Target: Windows Office Laptop / Developer Workstation
# Supported Models: Qwen2.5-VL-3B (Full), Moondream2 (Balanced), SmolVLM (Triage)
# ==============================================================================

param(
    [string]$Device = "",
    [switch]$NonInteractive
)

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Desktop VLM Lens: 1-Click Setup & Deployment Wizard" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# 1. Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python 3 is not found in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10 or newer from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    Exit 1
}

Write-Host "[1/4] Checking Python Environment: $($pythonCmd.Source)" -ForegroundColor Green

# 2. Virtual Environment Setup
if (-not (Test-Path "venv")) {
    Write-Host "[2/4] Creating Python virtual environment in .\venv..." -ForegroundColor Yellow
    python -m venv venv
} else {
    Write-Host "[2/4] Virtual environment .\venv already exists." -ForegroundColor Green
}

$venvPython = "$ScriptDir\venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -r requirements.txt --quiet
Write-Host "      Installed lightweight dependencies (Pillow, requests)." -ForegroundColor Green

# 3. Create folders & config
New-Item -ItemType Directory -Force -Path "bin" | Out-Null
New-Item -ItemType Directory -Force -Path "models" | Out-Null

if ($Device) {
    $cfg = @{
        "device" = $Device.ToLower()
        "cpu_threads" = $null
        "background_priority" = $true
        "port" = 8085
        "image_max_tokens" = 512
        "image_min_tokens" = 256
    }
    $cfg | ConvertTo-Json -Depth 3 | Set-Content -Path "config.json" -Encoding UTF8
    Write-Host "      Generated config.json (device: $Device, background_priority: true)." -ForegroundColor Green
}

# 4. Interactive Configuration & Acquisition
Write-Host ""
Write-Host "[3/4] Hardware & Model Configuration" -ForegroundColor Cyan
Write-Host "-------------------------------------------------------"

$binaryArgs = if ($Device) { @("scripts\fetch_binaries.py", "--device", $Device) } elseif ($NonInteractive) { @("scripts\fetch_binaries.py", "--auto") } else { @("scripts\fetch_binaries.py") }
& $venvPython @binaryArgs
Write-Host ""
$modelArgs = if ($NonInteractive) { @("scripts\fetch_models.py", "--model", "qwen2.5-vl-3b") } else { @("scripts\fetch_models.py") }
& $venvPython @modelArgs

# 5. Diagnostic Verification
Write-Host ""
Write-Host "[4/4] Verifying Environment & Capabilities" -ForegroundColor Cyan
Write-Host "-------------------------------------------------------"
& $venvPython "scripts\verify_environment.py"

# 6. MiniMax 2.7 / Claude Code MCP Registration Snippet
Write-Host ""
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "  Setup Complete! Ready to Deploy to MiniMax 2.7" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To give MiniMax 2.7 / Claude / agy eyes on your laptop, add this to your MCP config:" -ForegroundColor White
Write-Host ""
$mcpSnippet = @"
{
  "mcpServers": {
    "desktop-vlm-lens": {
      "command": "$($venvPython.Replace('\', '\\'))",
      "args": [
        "$($ScriptDir.Replace('\', '\\'))\\src\\server.py"
      ],
      "env": {
        "VLM_SERVER_URL": "http://127.0.0.1:8085"
      }
    }
  }
}
"@
Write-Host $mcpSnippet -ForegroundColor Yellow
Write-Host ""
Write-Host "To start the local background server anytime, run:" -ForegroundColor White
Write-Host "  .\launch_vlm_server.bat" -ForegroundColor Cyan
Write-Host ""
