[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Kt1Dir,

    [Parameter(Mandatory = $true)]
    [string]$Contents,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [ValidateRange(1, 64)]
    [int]$Workers = 8,

    [ValidateRange(1, 10000)]
    [int]$FilesPerShard = 1000,

    [ValidateSet('parquet', 'csv-gzip')]
    [string]$OutputFormat = 'parquet',

    [switch]$PreflightOnly,

    [switch]$SkipDiskCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-ExistingPathStrict {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    $resolved = Resolve-Path -LiteralPath $PathValue -ErrorAction Stop
    return $resolved.Path
}

function Get-BasePython {
    $candidates = @()

    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        $candidates += ,@('py.exe', '-3.11')
        $candidates += ,@('py.exe', '-3.12')
        $candidates += ,@('py.exe', '-3.13')
        $candidates += ,@('py.exe', '-3.10')
    }
    if (Get-Command python.exe -ErrorAction SilentlyContinue) {
        $candidates += ,@('python.exe')
    }

    foreach ($candidate in $candidates) {
        $exe = $candidate[0]
        $prefixArgs = @()
        if ($candidate.Count -gt 1) {
            $prefixArgs = $candidate[1..($candidate.Count - 1)]
        }
        try {
            $versionText = & $exe @prefixArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $versionText) {
                $parts = $versionText.Trim().Split('.')
                $major = [int]$parts[0]
                $minor = [int]$parts[1]
                if ($major -eq 3 -and $minor -ge 10 -and $minor -le 14) {
                    return [PSCustomObject]@{
                        Exe = $exe
                        PrefixArgs = $prefixArgs
                        Version = $versionText.Trim()
                    }
                }
            }
        }
        catch {
            continue
        }
    }

    throw 'Python 3.10-3.14 was not found. Install 64-bit Python 3.11 or 3.12, then run this launcher again.'
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$processorScript = Join-Path $scriptDir 'ednet_kt1_prepare.py'
if (-not (Test-Path -LiteralPath $processorScript -PathType Leaf)) {
    throw "Processor Python file not found: $processorScript"
}

$resolvedKt1 = Resolve-ExistingPathStrict -PathValue $Kt1Dir
$resolvedContents = Resolve-ExistingPathStrict -PathValue $Contents

if (-not (Test-Path -LiteralPath $resolvedKt1 -PathType Container)) {
    throw "-Kt1Dir is not a directory: $resolvedKt1"
}
if (-not (Test-Path -LiteralPath $resolvedContents)) {
    throw "-Contents path does not exist: $resolvedContents"
}

$outputFull = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Path $outputFull -Force | Out-Null

$basePython = Get-BasePython
Write-Host "Selected Python: $($basePython.Exe) $($basePython.PrefixArgs -join ' ') (Python $($basePython.Version))"

$venvDir = Join-Path $scriptDir '.venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    Write-Host "Creating isolated Python environment: $venvDir"
    $venvArgs = @($basePython.PrefixArgs) + @('-m', 'venv', $venvDir)
    & $basePython.Exe @venvArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment creation failed, exit code: $LASTEXITCODE"
    }
}

Write-Host 'Checking/installing pinned dependency pyarrow 25.0.0...'
& $venvPython -m pip install --disable-pip-version-check --only-binary=:all: --upgrade 'pyarrow==25.0.0'
if ($LASTEXITCODE -ne 0) {
    throw "pyarrow 25.0.0 installation failed, exit code: $LASTEXITCODE"
}

$pythonArgs = @(
    $processorScript,
    '--kt1-dir', $resolvedKt1,
    '--contents', $resolvedContents,
    '--output-dir', $outputFull,
    '--output-format', $OutputFormat,
    '--workers', $Workers.ToString(),
    '--files-per-shard', $FilesPerShard.ToString()
)
if ($PreflightOnly) {
    $pythonArgs += '--preflight-only'
}
if ($SkipDiskCheck) {
    $pythonArgs += '--skip-disk-check'
}

$commandRecord = [ordered]@{
    started_at = (Get-Date).ToString('o')
    kt1_dir = $resolvedKt1
    contents = $resolvedContents
    output_dir = $outputFull
    workers = $Workers
    files_per_shard = $FilesPerShard
    output_format = $OutputFormat
    preflight_only = [bool]$PreflightOnly
    python = $venvPython
    processor_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $processorScript).Hash.ToLowerInvariant()
    pyarrow_version = (& $venvPython -c "import pyarrow; print(pyarrow.__version__)")
}
$commandRecord | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $outputFull 'launcher_parameters.json') -Encoding UTF8

Write-Host ''
Write-Host 'Processing starts now. After interruption, the same command resumes after verified shards.'
Write-Host ''

& $venvPython @pythonArgs
$exitCode = $LASTEXITCODE

switch ($exitCode) {
    0 {
        Write-Host ''
        Write-Host 'Processing completed successfully.' -ForegroundColor Green
    }
    2 {
        Write-Warning 'Processing completed, but at least one learner file had a read error. Check the errors directory and manifest.json.'
    }
    default {
        Write-Host "Processor stopped with an error. Exit code: $exitCode. Completed shards remain and the same command can resume." -ForegroundColor Red
    }
}

exit $exitCode
