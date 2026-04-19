[CmdletBinding()]
param(
    [string]$ResourceGroup = "ca-rag-v2-rg",
    [string]$WebAppName = "caragv2api36972",
    [string]$StagingDir = ".deploy_backend",
    [string]$ZipPath = "deploy_backend.zip",
    [string]$StartupCommand = 'python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000',
    [switch]$PackageOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$stagingPath = Join-Path $repoRoot $StagingDir
$zipFilePath = Join-Path $repoRoot $ZipPath

if (Test-Path -LiteralPath $stagingPath) {
    Remove-Item -LiteralPath $stagingPath -Recurse -Force
}

if (Test-Path -LiteralPath $zipFilePath) {
    Remove-Item -LiteralPath $zipFilePath -Force
}

New-Item -ItemType Directory -Path $stagingPath | Out-Null

$itemsToCopy = @(
    "backend",
    "models",
    "prompts",
    "utils",
    "main.py",
    "startup.sh",
    "requirements.txt",
    ".env.example"
)

foreach ($item in $itemsToCopy) {
    $sourcePath = Join-Path $repoRoot $item
    Copy-Item -LiteralPath $sourcePath -Destination $stagingPath -Recurse -Force
}

Get-ChildItem -Path $stagingPath -Recurse -Directory -Filter "__pycache__" |
    Remove-Item -Recurse -Force

tar.exe -a -c -f $zipFilePath -C $stagingPath .

if ($PackageOnly) {
    return
}

az webapp config set `
    --resource-group $ResourceGroup `
    --name $WebAppName `
    --startup-file $StartupCommand | Out-Null

az resource update `
    --resource-group $ResourceGroup `
    --name "$WebAppName/config/web" `
    --resource-type Microsoft.Web/sites/config `
    --set properties.alwaysOn=true properties.healthCheckPath=/api/health | Out-Null

az webapp update `
    --resource-group $ResourceGroup `
    --name $WebAppName `
    --set httpsOnly=true | Out-Null

az webapp deploy `
    --resource-group $ResourceGroup `
    --name $WebAppName `
    --src-path $zipFilePath `
    --type zip `
    --clean true `
    --restart true
