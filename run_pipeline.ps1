[CmdletBinding()]
param(
    [int]$Port = 8084,
    [string]$CsvPath = "data/themes_assurance_emprunteur_100_fr.csv",
    [string]$OutDir = "outputs",
    [switch]$UseMock
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Normalize paths
if (-not [System.IO.Path]::IsPathRooted($CsvPath)) {
    $CsvPath = Join-Path $scriptDir $CsvPath
}
if (-not [System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $scriptDir $OutDir
}

# Ensure output directory exists
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

# Load .env if present (PowerShell syntax) when GEMINI_API_KEY is missing
$envFile = Join-Path $scriptDir ".env"
if (-not $env:GEMINI_API_KEY -and (Test-Path $envFile)) {
    try {
        Write-Host "Loading environment from $envFile ..." -ForegroundColor Cyan
        $dotenv = Get-Content -Raw -Path $envFile
        Invoke-Expression $dotenv
    } catch {
        Write-Warning "Failed to load .env: $($_.Exception.Message)"
    }
}

# Optional mock mode to avoid external API calls
if ($UseMock) {
    $env:MOCK_GEN = "1"
    Write-Host "Using MOCK_GEN=1 (no external API calls)" -ForegroundColor Yellow
} else {
    if (-not $env:GEMINI_API_KEY) {
        Write-Warning "GEMINI_API_KEY is not set. Real generation will likely fail. Run with -UseMock or set GEMINI_API_KEY."
    }
}

# Validate CSV path
if (-not (Test-Path $CsvPath)) {
    throw "CSV file not found: $CsvPath"
}

# Determine how to start Flask
$flaskArgs = "--app app run --port $Port"
if (Get-Command python -ErrorAction SilentlyContinue) {
    $exe = "python"
    $args = "-m flask $flaskArgs"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $exe = "py"
    $args = "-m flask $flaskArgs"
} else {
    $exe = "flask"
    $args = $flaskArgs
}

Write-Host "Starting Flask server: $exe $args (cwd=$scriptDir)" -ForegroundColor Cyan
$server = Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory $scriptDir -PassThru -WindowStyle Minimized
$serverPid = $server.Id

function Wait-ServerReady {
    param([string]$url, [int]$timeoutSec = 60)
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 3
            return $true
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    return $false
}

try {
    $base = "http://localhost:$Port"
    Write-Host "Waiting for server on $base ..." -ForegroundColor Cyan
    if (-not (Wait-ServerReady -url "$base/")) {
        throw "Server did not become ready within timeout."
    }
    Write-Host "Server is ready." -ForegroundColor Green

    # 1) Import themes
    $importBody = @{ csv_path = $CsvPath } | ConvertTo-Json -Depth 3
    Write-Host "Importing themes from $CsvPath ..." -ForegroundColor Cyan
    $importResp = Invoke-RestMethod -Uri "$base/themes/import" -Method Post -ContentType "application/json" -Body $importBody -TimeoutSec 120
    $importResp | ConvertTo-Json -Depth 10 | Out-File -FilePath (Join-Path $OutDir "response_import.json") -Encoding utf8
    Write-Host "Themes imported." -ForegroundColor Green

    # 2) Generate article from random theme
    Write-Host "Generating article from a random unconsumed theme ..." -ForegroundColor Cyan
    $nextResp = Invoke-RestMethod -Uri "$base/article/next" -Method Post -ContentType "application/json" -Body '{}' -TimeoutSec 1800
    $nextResp | ConvertTo-Json -Depth 100 | Out-File -FilePath (Join-Path $OutDir "response_article_next.json") -Encoding utf8

    $articleJson = $nextResp.article
    if (-not $articleJson) { throw "No 'article' object in /article/next response." }

    $articleJson | ConvertTo-Json -Depth 100 | Out-File -FilePath (Join-Path $OutDir "article_text.json") -Encoding utf8
    Write-Host "Structured article saved to $OutDir\article_text.json" -ForegroundColor Green

    # 3) Render HTML
    Write-Host "Rendering HTML ..." -ForegroundColor Cyan
    $renderBody = $articleJson | ConvertTo-Json -Depth 100
    $renderResp = Invoke-RestMethod -Uri "$base/render/html" -Method Post -ContentType "application/json" -Body $renderBody -TimeoutSec 600

    $htmlPath = Join-Path $OutDir "article.html"
    $renderResp.html | Out-File -FilePath $htmlPath -Encoding utf8
    ($renderResp.article | ConvertTo-Json -Depth 100) | Out-File -FilePath (Join-Path $OutDir "article_legacy.json") -Encoding utf8
    ($renderResp.social  | ConvertTo-Json -Depth 100) | Out-File -FilePath (Join-Path $OutDir "social.json") -Encoding utf8

    Write-Host "Rendered HTML saved to $htmlPath" -ForegroundColor Green
    Write-Host "Done." -ForegroundColor Green
}
catch {
    Write-Error $_
    exit 1
}
finally {
    if ($serverPid) {
        Write-Host "Stopping Flask server (PID $serverPid) ..." -ForegroundColor Cyan
        try { Stop-Process -Id $serverPid -Force -ErrorAction SilentlyContinue } catch {}
    }
}
