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
$AudioSourceDir = Join-Path $ProjectRoot "data\audio"
$OneDriveDir  = Join-Path $env:ONEDRIVE "WisdomLens_Backups"
$OneDriveAudioDir = Join-Path $OneDriveDir "audio"
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

# 4. Copy only audio files referenced by the current database to OneDrive
$AudioQuery = @"
SELECT DISTINCT audio_filename
FROM inquiries
WHERE audio_filename IS NOT NULL AND btrim(audio_filename) <> ''
ORDER BY audio_filename;
"@
$AudioFilenames = @(docker compose -f $ComposeFile exec -T postgres `
    psql -U $DbUser -d $DbName -At -c $AudioQuery)
if ($LASTEXITCODE -ne 0) {
    throw "Could not read audio filenames from the database."
}
$AudioFilenames = @($AudioFilenames | ForEach-Object { $_.Trim() } | Where-Object { $_ })

if ($AudioFilenames.Count -gt 0) {
    New-Item -ItemType Directory -Force -Path $OneDriveAudioDir | Out-Null

    foreach ($AudioFilename in $AudioFilenames) {
        if ([System.IO.Path]::GetFileName($AudioFilename) -ne $AudioFilename -or $AudioFilename -in @('.', '..')) {
            throw "Unexpected audio filename in database: $AudioFilename"
        }

        $AudioSourcePath = Join-Path $AudioSourceDir $AudioFilename
        if (-not (Test-Path -LiteralPath $AudioSourcePath -PathType Leaf)) {
            throw "Audio referenced by the database is missing: $AudioSourcePath"
        }

        $AudioDestinationPath = Join-Path $OneDriveAudioDir $AudioFilename
        Copy-Item -LiteralPath $AudioSourcePath -Destination $AudioDestinationPath -Force
    }

    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Copied $($AudioFilenames.Count) database-linked audio file(s) to OneDrive: $OneDriveAudioDir" -ForegroundColor Green
} else {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] No audio files referenced by the database; skipping audio backup."
}

# 5. Remove old local backups
$OldLocal = @(Get-ChildItem -Path $BackupsDir -Filter "wisdomlens_*.sql" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$RetainDays) })
if ($OldLocal.Count -gt 0) {
    $OldLocal | Remove-Item
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Removed $($OldLocal.Count) old local backup(s)."
}

# 6. Remove old OneDrive backups
$OldOneDrive = @(Get-ChildItem -Path $OneDriveDir -Filter "wisdomlens_*.sql" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$RetainOneDrive) })
if ($OldOneDrive.Count -gt 0) {
    $OldOneDrive | Remove-Item
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Removed $($OldOneDrive.Count) old OneDrive backup(s)."
}

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Backup finished successfully." -ForegroundColor Green
