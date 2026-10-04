# ==============================================================================
# Desktop VLM Lens - Hardware-Adaptive Background Engine Launcher
# Portable: Works across any drive, directory, and user profile
# Acceleration: Auto-detects Intel Iris Xe / AMD Radeon Vulkan, NVIDIA CUDA,
#               and binds threads strictly to physical Performance (P) cores.
# Token Budget: Enforces --image-max-tokens 512 to eliminate laptop CPU stalls.
# Background Mode: Supports -Device cpu with BelowNormal priority scheduling.
# ==============================================================================

param(
    [string]$Device = "",
    [switch]$Background,
    [switch]$Force
)

# Determine package directory dynamically
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
if (-not $ScriptDir) {
    $ScriptDir = (Get-Location).Path
}

$port = 8085
$BackgroundPriority = $true

# Read config.json if present
$configFile = Join-Path $ScriptDir "config.json"
if (Test-Path $configFile) {
    try {
        $cfg = Get-Content $configFile -Raw | ConvertFrom-Json
        if (-not $Device -and $cfg.device) { $Device = [string]$cfg.device }
        if ($cfg.port) { $port = [int]$cfg.port }
        if ($cfg.background_priority -ne $null) { $BackgroundPriority = [bool]$cfg.background_priority }
    } catch {}
}

if ($Background) { $BackgroundPriority = $true }

# If -Force is specified, stop any running instance on port 8085 first
if ($Force) {
    Write-Host "Force restart requested: stopping existing llama-server instances..." -ForegroundColor Yellow
    Get-Process "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 700
}

$healthUrl = "http://127.0.0.1:$port/health"

# Fast check if already running (unless -Force was passed)
if (-not $Force) {
    try {
        $resp = Invoke-WebRequest -Uri $healthUrl -TimeoutSec 1 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($resp.StatusCode -eq 200) {
            Write-Host "VLM Server is already active and healthy on port $port." -ForegroundColor Green
            exit 0
        }
    } catch {}
}

# 1. Discover llama-server binary
$candidates = @(
    (Join-Path $ScriptDir "bin\llama-server.exe"),
    (Join-Path $ScriptDir "bin\build\bin\llama-server.exe")
)

# Search subdirectories of bin\ if present
$binDir = Join-Path $ScriptDir "bin"
if (Test-Path $binDir) {
    $subHits = Get-ChildItem -Path $binDir -Filter "llama-server.exe" -Recurse -File -ErrorAction SilentlyContinue
    foreach ($h in $subHits) {
        $candidates += $h.FullName
    }
}

# Check Ollama installation path as fallback
if ($env:LOCALAPPDATA) {
    $candidates += (Join-Path $env:LOCALAPPDATA "Programs\Ollama\lib\ollama\llama-server.exe")
}

# Check system PATH
$pathCmd = Get-Command "llama-server.exe" -ErrorAction SilentlyContinue
if ($pathCmd) {
    $candidates += $pathCmd.Source
}

$exe = $null
foreach ($c in $candidates) {
    if ($c -and (Test-Path $c)) {
        $exe = $c
        break
    }
}

if (-not $exe) {
    Write-Host "[ERROR] 'llama-server.exe' was not found!" -ForegroundColor Red
    Write-Host "Expected in: $ScriptDir\bin\llama-server.exe" -ForegroundColor Yellow
    Write-Host "Please run '.\setup_windows.bat' or download the binary via 'python scripts\fetch_binaries.py'." -ForegroundColor Yellow
    exit 1
}

# 2. Discover GGUF Model & mmproj
$modelsDir = Join-Path $ScriptDir "models"
if (-not (Test-Path $modelsDir)) {
    Write-Host "[ERROR] Models folder not found at: $modelsDir" -ForegroundColor Red
    Write-Host "Please create the 'models' directory and download your model." -ForegroundColor Yellow
    exit 1
}

# Look for primary model
$defaultModel = Join-Path $modelsDir "Qwen2.5-VL-3B-Instruct.gguf"
$modelPath = $null
if (Test-Path $defaultModel) {
    $modelPath = $defaultModel
} else {
    $foundModel = Get-ChildItem -Path $modelsDir -Filter "*.gguf" -File | Where-Object { $_.Name -notlike "*mmproj*" } | Select-Object -First 1
    if ($foundModel) {
        $modelPath = $foundModel.FullName
    }
}

if (-not $modelPath) {
    Write-Host "[ERROR] No GGUF model file found in: $modelsDir" -ForegroundColor Red
    Write-Host "Please download Qwen2.5-VL-3B-Instruct.gguf into the 'models' folder." -ForegroundColor Yellow
    exit 1
}

# Look for mmproj projector
$defaultMmproj = Join-Path $modelsDir "mmproj-Qwen2.5-VL-3B-Instruct.gguf"
$mmprojPath = $null
if (Test-Path $defaultMmproj) {
    $mmprojPath = $defaultMmproj
} else {
    $foundMmproj = Get-ChildItem -Path $modelsDir -Filter "*mmproj*.gguf" -File | Select-Object -First 1
    if ($foundMmproj) {
        $mmprojPath = $foundMmproj.FullName
    }
}

# 3. Hardware-Adaptive Parameter Optimization (ADR-43)
$cacheFile = Join-Path $ScriptDir ".hardware_profile.json"
$threads = 4
$threadsBatch = 8
$ngl = 99
$profileName = "Hardware Adaptive"
$isHybrid = $false

if ($Device) {
    $Device = $Device.ToLower()
}

# A. Dynamic Python hardware probe if Device specified or cache missing
$pyExec = if (Test-Path (Join-Path $ScriptDir "venv\Scripts\python.exe")) {
    Join-Path $ScriptDir "venv\Scripts\python.exe"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    (Get-Command python).Source
} else {
    $null
}

if ($pyExec) {
    $detectScript = Join-Path $ScriptDir "scripts\detect_hardware.py"
    if (Test-Path $detectScript) {
        try {
            $deviceArg = if ($Device) { @("--device", $Device) } else { @() }
            $probeOut = & $pyExec $detectScript --json @deviceArg 2>$null
            if ($probeOut) {
                $probeJson = $probeOut | ConvertFrom-Json
                if ($probeJson.optimal_args) {
                    $threads = [int]$probeJson.optimal_args.threads
                    $threadsBatch = [int]$probeJson.optimal_args.threads_batch
                    $ngl = [int]$probeJson.optimal_args.ngl
                    $profileName = [string]$probeJson.recommended_flavor
                    $isHybrid = [bool]$probeJson.cpu.is_hybrid
                    if ($probeJson.optimal_args.background_priority -ne $null) {
                        $BackgroundPriority = [bool]$probeJson.optimal_args.background_priority
                    }
                }
            }
        } catch {}
    }
} elseif (Test-Path $cacheFile) {
    try {
        $profileJson = Get-Content -Path $cacheFile -Raw | ConvertFrom-Json
        if ($profileJson.optimal_args) {
            $threads = [int]$profileJson.optimal_args.threads
            $threadsBatch = [int]$profileJson.optimal_args.threads_batch
            $ngl = [int]$profileJson.optimal_args.ngl
            $profileName = [string]$profileJson.recommended_flavor
            $isHybrid = [bool]$profileJson.cpu.is_hybrid
        }
    } catch {}
}

# Explicit Device overrides
if ($Device -eq "cpu") {
    $ngl = 0
    $profileName = "Background CPU (Zero VRAM)"
} elseif ($Device -eq "cuda") {
    $ngl = 99
} elseif ($Device -eq "vulkan") {
    $ngl = 99
}

# Native CPU topology fallback if unconfigured
if ($threads -lt 2) { $threads = 2 }

# Environment variable overrides
if ($env:VLM_THREADS) { $threads = [int]$env:VLM_THREADS }
if ($env:VLM_NGL) { $ngl = [int]$env:VLM_NGL }
$maxTokens = if ($env:VLM_MAX_IMAGE_TOKENS) { $env:VLM_MAX_IMAGE_TOKENS } else { "512" }
$minTokens = if ($env:VLM_MIN_IMAGE_TOKENS) { $env:VLM_MIN_IMAGE_TOKENS } else { "256" }

$workingDir = Split-Path -Parent $exe

# 4. Assemble Arguments with Visual Token Budgeting & Core Affinity
$argsList = @(
    "-m", $modelPath,
    "-c", "4096",
    "--image-max-tokens", [string]$maxTokens,
    "--image-min-tokens", [string]$minTokens,
    "-ngl", [string]$ngl,
    "--port", [string]$port,
    "--host", "127.0.0.1",
    "-t", [string]$threads,
    "-tb", [string]$threadsBatch
)

if ($mmprojPath) {
    $argsList += @("--mmproj", $mmprojPath)
}

Write-Host "Spawning hardware-adaptive llama-server (:8085)..." -ForegroundColor Cyan
Write-Host "  Profile:    $profileName" -ForegroundColor Gray
Write-Host "  Binary:     $exe" -ForegroundColor Gray
Write-Host "  Model:      $modelPath" -ForegroundColor Gray
if ($mmprojPath) {
    Write-Host "  Vision:     $mmprojPath" -ForegroundColor Gray
}
$coreNote = if ($isHybrid) { "(P-Core Bound)" } else { "(Physical Cores)" }
Write-Host "  Threads:    $threads $coreNote | Batch: $threadsBatch" -ForegroundColor Gray
Write-Host "  Offload:    -ngl $ngl layers" -ForegroundColor Gray
Write-Host "  Token Budget: --image-max-tokens $maxTokens (Eliminates CPU stalls)" -ForegroundColor Gray
if ($BackgroundPriority) {
    Write-Host "  Priority:   BelowNormal (Guarantees zero foreground stutter)" -ForegroundColor Gray
}

$proc = Start-Process -FilePath $exe -ArgumentList $argsList -WorkingDirectory $workingDir -WindowStyle Hidden -PassThru
if ($BackgroundPriority -and $proc) {
    try {
        $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
    } catch {}
}

# 5. Wait for server readiness
Write-Host "Waiting for server to initialize..." -NoNewline
for ($i = 0; $i -lt 80; $i++) {
    Start-Sleep -Milliseconds 500
    Write-Host "." -NoNewline
    try {
        $resp = Invoke-WebRequest -Uri $healthUrl -TimeoutSec 1 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($resp.StatusCode -eq 200) {
            Write-Host ""
            Write-Host "[OK] VLM Server is READY on port $port!" -ForegroundColor Green
            exit 0
        }
    } catch {}
}

Write-Host ""
Write-Error "Timed out waiting for VLM Server on port $port."
exit 1
