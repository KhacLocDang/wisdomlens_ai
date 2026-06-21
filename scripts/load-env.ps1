# Load key=value pairs from the project .env into the current process.
function Import-ProjectDotEnv {
    param([string]$ProjectRoot)

    $envFile = Join-Path $ProjectRoot ".env"
    if (-not (Test-Path $envFile)) {
        Write-Error "Missing .env file. Copy .env.example to .env and set POSTGRES_* values."
        exit 1
    }

    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#=\s]+)\s*=\s*(.*)\s*$') {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2].Trim() -Force
        }
    }
}

function Get-RequiredEnv {
    param([string]$Name)

    $value = [Environment]::GetEnvironmentVariable($Name, "Process")
    if (-not $value) {
        Write-Error "Missing $Name in .env (see .env.example)."
        exit 1
    }
    return $value
}
