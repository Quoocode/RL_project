param(
    [string]$ProjectId
)

$ErrorActionPreference = "Stop"

if (-not $ProjectId) {
    throw "ProjectId is required"
}

gcloud config set project $ProjectId | Out-Null

$regions = @(
    "asia-southeast1",
    "asia-east1",
    "asia-east2",
    "us-central1",
    "us-east1",
    "us-west1"
)

$requiredCpu = 20

foreach ($region in $regions) {
    Write-Host "`n=== $region ==="

    $json = gcloud compute regions describe $region `
        --project=$ProjectId `
        --format=json | ConvertFrom-Json

    $e2 = $json.quotas | Where-Object { $_.metric -eq "E2_CPUS" }
    $cpu = $json.quotas | Where-Object { $_.metric -eq "CPUS" }
    $addr = $json.quotas | Where-Object { $_.metric -eq "IN_USE_ADDRESSES" }

    $quota = if ($e2) { $e2 } else { $cpu }
    $available = [double]$quota.limit - [double]$quota.usage

    Write-Host "CPU metric : $($quota.metric)"
    Write-Host "Limit      : $($quota.limit)"
    Write-Host "Usage      : $($quota.usage)"
    Write-Host "Available  : $available"
    if ($addr) {
        Write-Host "Addresses  : limit=$($addr.limit), usage=$($addr.usage)"
    }

    if ($available -ge $requiredCpu) {
        Write-Host "✅ OK: đủ quota cho 20 vCPU"
    } else {
        Write-Host "❌ Không đủ quota"
    }
}