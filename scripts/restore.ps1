# WisdomLens AI — Restore script
# Restores database from a SQL backup file.
#
# Usage:
#   .\scripts\restore.ps1 -BackupFile ".\backups\wisdomlens_2026-06-17_0800.sql"

param(
    [Parameter(Mandatory = $true)]
    [string]$BackupFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "load-env.ps1")

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $ProjectRoot "docker-compose.yml"

Import-ProjectDotEnv -ProjectRoot $ProjectRoot
$DbUser = Get-RequiredEnv -Name "POSTGRES_USER"
$DbName = Get-RequiredEnv -Name "POSTGRES_DB"

if (-not (Test-Path $BackupFile)) {
    Write-Error "Backup file not found: $BackupFile"
    exit 1
}

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Restoring from: $BackupFile" -ForegroundColor Cyan
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] WARNING: Existing data will be overwritten. Continue? (y/N)" -ForegroundColor Yellow
$confirm = Read-Host
if ($confirm -ne "y" -and $confirm -ne "Y") {
    Write-Host "Aborted."
    exit 0
}

Get-Content $BackupFile | docker compose -f $ComposeFile exec -T postgres `
    psql -U $DbUser -d $DbName

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Restore complete." -ForegroundColor Green
