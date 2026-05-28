param(
    [switch]$IncludeLocalDeploy
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

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
            Write-Host "Stopping stale process pid=$($process.ProcessId)"
            Stop-Process -Id ([int]$process.ProcessId) -Force -ErrorAction SilentlyContinue
        }
    }
}

$pidFiles = @(
    ".logs\stockpredictor-api-8000.pid",
    ".logs\stockpredictor-dashboard-8501.pid"
)

if ($IncludeLocalDeploy) {
    $pidFiles += ".logs\localdeploy-8100.pid"
}

foreach ($pidFile in $pidFiles) {
    if (-not (Test-Path -LiteralPath $pidFile)) {
        continue
    }
    $processId = Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue
    if (-not $processId) {
        continue
    }
    $process = Get-Process -Id ([int]$processId) -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "Stopping $($process.ProcessName) pid=$processId"
        Stop-Process -Id ([int]$processId) -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}

Stop-MatchingProcesses @($ProjectRoot, "stockpredictor.cli api")
Stop-MatchingProcesses @($ProjectRoot, "stockpredictor.cli dashboard")
Stop-MatchingProcesses @($ProjectRoot, "streamlit run", "stockpredictor\ui\dashboard.py")
if ($IncludeLocalDeploy) {
    Stop-MatchingProcesses @("LocalDeploy", "api_server.py")
}

Write-Host "StockPredictor local services stopped. Use -IncludeLocalDeploy to also stop LocalDeploy."
