[CmdletBinding()]
param(
    [string]$ResourceGroup = "ca-rag-v2-rg",
    [string]$StorageAccountName = "caragv2fe36972",
    [string]$ApiBaseUrl = "https://caragv2api36972.azurewebsites.net",
    [string]$StorageAccountKey = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$frontendPath = Join-Path $repoRoot "frontend"
$previousApiBaseUrl = $env:NEXT_PUBLIC_API_BASE_URL

try {
    Set-Location $frontendPath
    $env:NEXT_PUBLIC_API_BASE_URL = $ApiBaseUrl

    npm ci
    npm run build
}
finally {
    if ($null -eq $previousApiBaseUrl) {
        Remove-Item Env:NEXT_PUBLIC_API_BASE_URL -ErrorAction SilentlyContinue
    }
    else {
        $env:NEXT_PUBLIC_API_BASE_URL = $previousApiBaseUrl
    }
}

$storageKey = $StorageAccountKey
if (-not $storageKey) {
    $storageKey = az storage account keys list `
        --account-name $StorageAccountName `
        --resource-group $ResourceGroup `
        --query "[0].value" `
        -o tsv
}

az storage blob service-properties update `
    --account-name $StorageAccountName `
    --account-key $storageKey `
    --static-website `
    --index-document index.html `
    --404-document 404.html

az storage blob delete-batch `
    --account-name $StorageAccountName `
    --account-key $storageKey `
    --source '$web'

az storage blob upload-batch `
    --account-name $StorageAccountName `
    --account-key $storageKey `
    --destination '$web' `
    --source (Join-Path $frontendPath "out") `
    --overwrite true
