param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$projectRoot = $PSScriptRoot
$runtimeRoot = Join-Path $projectRoot 'runtime'
$modelRoot = Join-Path $projectRoot 'models'
$downloadRoot = Join-Path $runtimeRoot 'downloads'
$modelName = 'qwen2.5-1.5b-instruct-q4_k_m.gguf'
$modelHash = '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
$runtimeHash = '5828cccc7261b14607d23de3144f35fac4249d9fd207e13bff5e31dd8ae39d56'
$archiveName = 'llama-b10826-bin-win-cpu-x64.zip'

function Test-Hash([string]$Path, [string]$Expected) {
    return ((Test-Path -LiteralPath $Path -PathType Leaf) -and
        ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Expected))
}

function Get-VerifiedDownload([string]$Url, [string]$Destination, [string]$Expected) {
    if (Test-Hash $Destination $Expected) {
        Write-Host "Already verified: $([IO.Path]::GetFileName($Destination))"
        return
    }
    if (Test-Path -LiteralPath $Destination) {
        throw "An existing file has an unexpected checksum: $Destination. Move it aside before retrying; setup will not overwrite it."
    }
    $partial = "$Destination.partial"
    Write-Host "Downloading $([IO.Path]::GetFileName($Destination)) ..."
    & curl.exe --fail --location --retry 3 --connect-timeout 30 --speed-limit 1024 --speed-time 120 --continue-at - --output $partial $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download interrupted. Run setup again to resume: $partial"
    }
    if (-not (Test-Hash $partial $Expected)) {
        throw "Checksum verification failed: $partial. Move this partial download aside before retrying."
    }
    Move-Item -LiteralPath $partial -Destination $Destination
}

$modelPath = Join-Path $modelRoot $modelName
$serverPath = Join-Path $runtimeRoot 'llama\llama-server.exe'
if ($CheckOnly) {
    if (-not (Test-Hash $modelPath $modelHash)) { throw 'The local model is missing or damaged. Run setup_chatbot.ps1.' }
    if (-not (Test-Path -LiteralPath $serverPath -PathType Leaf)) { throw 'The local runtime is missing. Run setup_chatbot.ps1.' }
    & $serverPath --version
    if ($LASTEXITCODE -ne 0) { throw 'The local runtime could not start.' }
    Write-Host 'Local chatbot files are ready.'
    exit 0
}

if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
    throw 'This setup package requires an x64 Windows PC.'
}
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw 'curl.exe is required. It is included with current Windows 10 and Windows 11.'
}
foreach ($directory in @($runtimeRoot, $modelRoot, $downloadRoot)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

Write-Host 'Setting up local chat. Initial download: approximately 1.14 GB.'
Write-Host 'Downloads come from the official llama.cpp and Qwen projects and are checked with SHA-256.'
$archivePath = Join-Path $downloadRoot $archiveName
Get-VerifiedDownload "https://github.com/ggml-org/llama.cpp/releases/download/b10826/$archiveName" $archivePath $runtimeHash
Get-VerifiedDownload "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/$modelName" $modelPath $modelHash

# The archive is pinned and verified before any executable is extracted or run.
$llamaRoot = Join-Path $runtimeRoot 'llama'
Expand-Archive -LiteralPath $archivePath -DestinationPath $llamaRoot -Force
if (-not (Test-Path -LiteralPath $serverPath -PathType Leaf)) {
    throw "The verified runtime archive does not contain the expected executable: $serverPath"
}
& $serverPath --version
if ($LASTEXITCODE -ne 0) { throw 'The runtime was downloaded, but could not start on this computer.' }
Write-Host 'Setup complete. Double-click Start Chatbot.cmd to chat.'
