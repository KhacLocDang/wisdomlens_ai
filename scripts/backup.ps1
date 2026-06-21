# WisdomLens AI — Backup script
# Dumps PostgreSQL database and copies to OneDrive.
#
# Usage:
#   .\scripts\backup.ps1
#
# Schedule (Task Scheduler) — replace <project-root> with your clone path:
#   Action: powershell.exe -ExecutionPolicy Bypass -File "<project-root>\scripts\backup.ps1"
#   Trigger: Daily at desired time

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "load-env.ps1")

# ── Configuration ────────────────────────────────────────────────────────────
$ProjectRoot  = Split-Path -Parent $PSScriptRoot
$BackupsDir   = Join-Path $ProjectRoot "backups"
$OneDriveDir  = Join-Path $env:ONEDRIVE "WisdomLens_Backups"
$RetainDays   = 14          # delete local backups older than this
$RetainOneDrive = 30        # delete OneDrive backups older than this
$ComposeFile  = Join-Path $ProjectRoot "docker-compose.yml"

Import-ProjectDotEnv -ProjectRoot $ProjectRoot
$DbUser = Get-RequiredEnv -Name "POSTGRES_USER"
$DbName = Get-RequiredEnv -Name "POSTGRES_DB"
# ─────────────────────────────────────────────────────────────────────────────

$Timestamp  = Get-Date -Format "yyyy-MM-dd_HHmm"
$FileName   = "wisdomlens_$Timestamp.sql"
$LocalPath  = Join-Path $BackupsDir $FileName

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting backup..." -ForegroundColor Cyan

# 1. Ensure directories exist
New-Item -ItemType Directory -Force -Path $BackupsDir   | Out-Null
New-Item -ItemType Directory -Force -Path $OneDriveDir  | Out-Null

# 2. Dump database (plain SQL — readable and portable)
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Dumping database..."
docker compose -f $ComposeFile exec -T postgres `
    pg_dump -U $DbUser -d $DbName --no-password `
    | Out-File -FilePath $LocalPath -Encoding utf8

$Size = (Get-Item $LocalPath).Length / 1KB
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Dump complete: $FileName ($([Math]::Round($Size, 1)) KB)"

# 3. Copy to OneDrive
$OneDrivePath = Join-Path $OneDriveDir $FileName
Copy-Item -Path $LocalPath -Destination $OneDrivePath
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Copied to OneDrive: $OneDrivePath" -ForegroundColor Green

# 4. Remove old local backups
$OldLocal = Get-ChildItem -Path $BackupsDir -Filter "wisdomlens_*.sql" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$RetainDays) }
if ($OldLocal) {
    $OldLocal | Remove-Item
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Removed $($OldLocal.Count) old local backup(s)."
}

# 5. Remove old OneDrive backups
$OldOneDrive = Get-ChildItem -Path $OneDriveDir -Filter "wisdomlens_*.sql" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$RetainOneDrive) }
if ($OldOneDrive) {
    $OldOneDrive | Remove-Item
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Removed $($OldOneDrive.Count) old OneDrive backup(s)."
}

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Backup finished successfully." -ForegroundColor Green
