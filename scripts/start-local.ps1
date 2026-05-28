param(
    [string]$ConfigPath = "configs/default.yaml",
    [string]$LocalDeployPath = "..\LocalDeploy",
    [int]$ApiPort = 8000,
    [int]$DashboardPort = 8501,
    [int]$LocalDeployPort = 8100,
    [switch]$NoBrowser,
    [switch]$ReuseExisting,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-Http {
    param(
        [string]$Url,
        [int]$TimeoutSec = 10
    )
    try {
        Invoke-RestMethod -Uri $Url -TimeoutSec $TimeoutSec | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Wait-Http {
    param(
        [string]$Url,
        [int]$Seconds = 60
    )
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Http $Url 3) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Ensure-StockPredictor {
    $python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if ((Test-Path -LiteralPath $python) -or $SkipInstall) {
        if (Test-Path -LiteralPath $python) {
            return $python
        }
        return "python"
    }

    Write-Step "Creating StockPredictor .venv"
    python -m venv .venv
    & $python -m pip install -U pip
    & $python -m pip install -e ".[dev]"
    return $python
}

function Ensure-Config {
    $resolved = Join-Path $ProjectRoot $ConfigPath
    if (-not (Test-Path -LiteralPath $resolved)) {
        $example = Join-Path $ProjectRoot "configs\default.example.yaml"
        if (-not (Test-Path -LiteralPath $example)) {
            throw "Config not found: $ConfigPath, and configs/default.example.yaml is missing."
        }
        Write-Step "Creating local config at $ConfigPath"
        Copy-Item -LiteralPath $example -Destination $resolved
    }
}

function Stop-MatchingProcesses {
    param([string[]]$Patterns)
    $currentPid = $PID
    $processes = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue
    foreach ($process in $processes) {
        $command = [string]$process.CommandLine
        if (-not $command -or [int]$process.ProcessId -eq $currentPid) {
            continue
        }
        $matches = $true
        foreach ($pattern in $Patterns) {
            if ($command -notlike "*$pattern*") {
                $matches = $false
                break
            }
        }
        if ($matches) {
            Write-Host "Stopping stale process pid=$($process.ProcessId)" -ForegroundColor Yellow
            Stop-Process -Id ([int]$process.ProcessId) -Force -ErrorAction SilentlyContinue
        }
    }
}

function Start-ServiceIfNeeded {
    param(
        [string]$Name,
        [string]$HealthUrl,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory,
        [hashtable]$Environment = @{},
        [int]$WaitSeconds = 60,
        [int]$HealthTimeoutSec = 10
    )

    if (Test-Http $HealthUrl $HealthTimeoutSec) {
        Write-Host "$Name already running at $HealthUrl" -ForegroundColor Green
        return
    }

    Write-Step "Starting $Name"
    New-Item -ItemType Directory -Force -Path ".logs" | Out-Null
    $out = Join-Path (Resolve-Path ".logs") "$Name.out.log"
    $err = Join-Path (Resolve-Path ".logs") "$Name.err.log"
    foreach ($key in $Environment.Keys) {
        Set-Item -Path "Env:$key" -Value ([string]$Environment[$key])
    }
    $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
    $process.Id | Set-Content -Path ".logs\$Name.pid"

    if (-not (Wait-Http $HealthUrl $WaitSeconds)) {
        Write-Host "$Name did not become ready. Last stderr lines:" -ForegroundColor Yellow
        Get-Content $err -Tail 80 -ErrorAction SilentlyContinue
        throw "$Name failed readiness check at $HealthUrl"
    }
    Write-Host "$Name ready at $HealthUrl" -ForegroundColor Green
}

Ensure-Config
$python = Ensure-StockPredictor
$logs = Join-Path $ProjectRoot ".logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

if (-not $ReuseExisting) {
    Write-Step "Stopping stale StockPredictor API/dashboard processes"
    Stop-MatchingProcesses @($ProjectRoot, "stockpredictor.cli api")
    Stop-MatchingProcesses @($ProjectRoot, "stockpredictor.cli dashboard")
    Stop-MatchingProcesses @($ProjectRoot, "streamlit run", "stockpredictor\ui\dashboard.py")
}

$localDeployRoot = Resolve-Path -LiteralPath (Join-Path $ProjectRoot $LocalDeployPath) -ErrorAction SilentlyContinue
if ($localDeployRoot) {
    $localDeployPython = Join-Path $localDeployRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $localDeployPython)) {
        $localDeployPython = "python"
    }
    Start-ServiceIfNeeded `
    -Name "localdeploy-$LocalDeployPort" `
        -HealthUrl "http://127.0.0.1:$LocalDeployPort/health" `
        -FilePath $localDeployPython `
        -ArgumentList @("api_server.py") `
        -WorkingDirectory $localDeployRoot `
    -Environment @{
        "API_HOST" = "127.0.0.1"
        "API_PORT" = $LocalDeployPort
        "CONFIG_PATH" = "config.json"
        "DEFAULT_MODEL_PROFILE" = "qwen3vl_8b_ollama"
    } `
        -WaitSeconds 90 `
        -HealthTimeoutSec 15
}
else {
    Write-Host "LocalDeploy path not found: $LocalDeployPath. News LLM will require a running compatible endpoint." -ForegroundColor Yellow
}

Start-ServiceIfNeeded `
    -Name "stockpredictor-api-$ApiPort" `
    -HealthUrl "http://127.0.0.1:$ApiPort/health" `
    -FilePath $python `
    -ArgumentList @("-m", "stockpredictor.cli", "api", "--config", $ConfigPath, "--host", "127.0.0.1", "--port", "$ApiPort") `
    -WorkingDirectory $ProjectRoot `
    -WaitSeconds 60

Start-ServiceIfNeeded `
    -Name "stockpredictor-dashboard-$DashboardPort" `
    -HealthUrl "http://127.0.0.1:$DashboardPort" `
    -FilePath $python `
    -ArgumentList @("-m", "stockpredictor.cli", "dashboard", "--config", $ConfigPath, "--server-port", "$DashboardPort") `
    -WorkingDirectory $ProjectRoot `
    -WaitSeconds 60

Write-Host ""
Write-Host "StockPredictor stack is ready:" -ForegroundColor Green
Write-Host "  Dashboard:   http://127.0.0.1:$DashboardPort"
Write-Host "  API:         http://127.0.0.1:$ApiPort"
Write-Host "  LocalDeploy: http://127.0.0.1:$LocalDeployPort"
Write-Host "  Logs:        $logs"
Write-Host ""
Write-Host "Stop later with: .\scripts\stop-local.ps1"

if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$DashboardPort"
}
