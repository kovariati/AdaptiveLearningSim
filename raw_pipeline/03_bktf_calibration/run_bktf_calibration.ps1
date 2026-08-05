[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputBundle,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Resolve-FullPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,
        [switch]$MustExist
    )

    if ($MustExist) {
        return (Resolve-Path -LiteralPath $PathValue).Path
    }
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path -Path (Get-Location).Path -ChildPath $PathValue))
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host "`n[$Label]" -ForegroundColor Cyan
    & $Command
    $code = $LASTEXITCODE
    if ($null -eq $code) { $code = 0 }
    if ($code -ne 0) {
        throw "$Label failed with exit code $code."
    }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$InputPath = Resolve-FullPath -PathValue $InputBundle -MustExist
$OutputPath = Resolve-FullPath -PathValue $OutputDir
$OutputParent = Split-Path -Parent $OutputPath
if (-not (Test-Path -LiteralPath $OutputParent -PathType Container)) {
    New-Item -ItemType Directory -Path $OutputParent -Force | Out-Null
}

$InputParent = Split-Path -Parent $InputPath
$PythonCandidates = @(
    (Join-Path $InputParent '.venv\Scripts\python.exe'),
    (Join-Path (Split-Path -Parent $ScriptDir) '.venv\Scripts\python.exe'),
    (Join-Path $ScriptDir '.venv\Scripts\python.exe')
)

$PythonExe = $null
foreach ($Candidate in $PythonCandidates) {
    if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
        $PythonExe = $Candidate
        break
    }
}

if ($null -eq $PythonExe) {
    $LocalVenv = Join-Path $ScriptDir '.venv'
    Write-Host "No reusable Python environment was found. Creating: $LocalVenv" -ForegroundColor Yellow
    $PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        & $PyLauncher.Source -3.11 -m venv $LocalVenv
    } else {
        $PythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($null -eq $PythonCommand) {
            throw 'Python 3.11 x64 was not found.'
        }
        & $PythonCommand.Source -m venv $LocalVenv
    }
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create the Python virtual environment.'
    }
    $PythonExe = Join-Path $LocalVenv 'Scripts\python.exe'
}

Write-Host "Python: $PythonExe" -ForegroundColor Green
& $PythonExe -c "import sys; print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw 'The selected Python interpreter cannot be started.' }

& $PythonExe -c "import numpy, scipy, pyarrow; print('numpy=' + numpy.__version__); print('scipy=' + scipy.__version__); print('pyarrow=' + pyarrow.__version__)"
$DependencyCode = $LASTEXITCODE
if ($DependencyCode -ne 0) {
    Write-Host 'Installing the tested binary dependencies...' -ForegroundColor Yellow
    Invoke-Checked -Label 'Pip bootstrap' -Command {
        & $PythonExe -m pip install --upgrade pip
    }
    Invoke-Checked -Label 'Dependency installation' -Command {
        & $PythonExe -m pip install 'numpy==2.3.5' 'scipy==1.17.0' 'pyarrow==25.0.0'
    }
}

$Program = Join-Path $ScriptDir 'adaptivelearningsim_bktf_calibrate.py'
if (-not (Test-Path -LiteralPath $Program -PathType Leaf)) {
    throw "Calibration program is missing: $Program"
}

Invoke-Checked -Label 'Python syntax check' -Command {
    & $PythonExe -m py_compile $Program
}

Invoke-Checked -Label 'Parquet, gradient, and parameter-recovery self-test' -Command {
    & $PythonExe $Program --self-test
}

Write-Host "`nStarting the complete 20-skill BKT/BKT-F calibration." -ForegroundColor Green
Write-Host 'Completed skill checkpoints are preserved. Run the same command after an interruption.' -ForegroundColor Green

& $PythonExe $Program --input-bundle $InputPath --output-dir $OutputPath
$ExitCode = $LASTEXITCODE

switch ($ExitCode) {
    0 {
        Write-Host "`nBKT/BKT-F CALIBRATION: PASS" -ForegroundColor Green
        Write-Host 'Upload this file:' -ForegroundColor Green
        Write-Host (Join-Path $OutputPath 'results\AdaptiveLearningSim_BKTF_calibration_bundle.zip') -ForegroundColor White
        exit 0
    }
    2 {
        Write-Host "`nBKT/BKT-F CALIBRATION: CONFIGURATION OR DATA ERROR" -ForegroundColor Red
        exit 2
    }
    130 {
        Write-Host "`nBKT/BKT-F CALIBRATION: INTERRUPTED. Completed skill checkpoints were preserved." -ForegroundColor Yellow
        exit 130
    }
    default {
        Write-Host "`nBKT/BKT-F CALIBRATION: UNEXPECTED ERROR (exit code $ExitCode)" -ForegroundColor Red
        exit $ExitCode
    }
}
