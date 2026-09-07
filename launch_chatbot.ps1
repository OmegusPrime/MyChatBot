param([switch]$Terminal)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonCandidates = @((Join-Path $projectRoot '.venv\Scripts\python.exe'))
foreach ($commandName in @('python', 'py')) {
    $candidate = Get-Command $commandName -ErrorAction SilentlyContinue
    if ($candidate) { $pythonCandidates += $candidate.Source }
}
$pythonExecutable = $null
foreach ($candidate in $pythonCandidates) {
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
    try {
        & $candidate -c 'import sys, tkinter; assert sys.version_info >= (3, 11)' 2>$null
        if ($LASTEXITCODE -eq 0) { $pythonExecutable = $candidate; break }
    } catch { continue }
}
if (-not $pythonExecutable) {
    throw 'A working Python 3.11 or newer with Tkinter is required. Install Python from python.org, or recreate the project .venv with a working Python installation.'
}
if ($Terminal) {
    & $pythonExecutable (Join-Path $projectRoot 'chat_cli.py')
    exit $LASTEXITCODE
}
# Run synchronously so startup failures remain visible in the launcher window.
& $pythonExecutable (Join-Path $projectRoot 'chat_gui.py')
exit $LASTEXITCODE
