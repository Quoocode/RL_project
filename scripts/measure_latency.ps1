param(
    [string]$ClusterStatePath = "cluster_state.json",
    [string]$OutputJson = "real_node_latency_matrix.json",
    [string]$OutputCsv = "real_node_latency_matrix.csv",
    [int]$PingCount = 10
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ClusterStatePath)) {
    throw "cluster_state.json not found. Run provision_gke.ps1 first."
}

$state = Get-Content $ClusterStatePath -Raw | ConvertFrom-Json
$namespace = $state.namespace
$nodeOrder = @($state.node_order | Sort-Object index)

Write-Host "Namespace: $namespace"
Write-Host "Checking latency test pods..."

$pods = @()

foreach ($node in $nodeOrder) {
    $idx = [int]$node.index
    $podName = "netshoot-node$idx"

    $phase = kubectl get pod $podName `
        -n $namespace `
        -o jsonpath="{.status.phase}"

    if ($phase -ne "Running") {
        throw "Pod $podName is not Running. Current phase: $phase. Check: kubectl get pods -n $namespace -l app=latency-test -o wide"
    }

    $podIp = kubectl get pod $podName `
        -n $namespace `
        -o jsonpath="{.status.podIP}"

    if (-not $podIp) {
        throw "Pod $podName has no IP."
    }

    $pods += [ordered]@{
        index = $idx
        pod = $podName
        ip = $podIp
        node = $node.name
        role = $node.role
    }

    Write-Host "  Node $idx [$($node.role)] $podName -> $podIp"
}

$n = $pods.Count
$matrix = @()
$csvRows = @()
$csvRows += "src,dst,avg_ms"

for ($i = 0; $i -lt $n; $i++) {
    $row = @()

    for ($j = 0; $j -lt $n; $j++) {
        if ($i -eq $j) {
            $row += 0.0
            $csvRows += "$i,$j,0.000"
            continue
        }

        $srcPod = $pods[$i].pod
        $dstIp = $pods[$j].ip

        Write-Host "Pinging $i -> $j  ($srcPod -> $dstIp)"

        $out = kubectl exec -n $namespace $srcPod -- ping -c $PingCount $dstIp 2>$null
        $line = $out | Select-String "rtt|round-trip"

        if ($line) {
            $text = $line.ToString()
            $parts = ($text -split "=")[1].Trim() -split "/"
            $avg = [double]$parts[1]

            $row += $avg
            $csvRows += "$i,$j,$avg"
        }
        else {
            Write-Host "  Warning: ping failed for $i -> $j, using 1.0 ms"
            $row += 1.0
            $csvRows += "$i,$j,ERROR"
        }
    }

    $matrix += ,$row
}

$result = [ordered]@{
    namespace = $namespace
    ping_count = $PingCount
    node_order = $nodeOrder
    pods = $pods
    latency_ms = $matrix
}

$result | ConvertTo-Json -Depth 20 | Out-File -Encoding utf8 $OutputJson
$csvRows | Out-File -Encoding utf8 $OutputCsv

Write-Host ""
Write-Host "Saved $OutputJson"
Write-Host "Saved $OutputCsv"
Write-Host ""
Write-Host "Latency matrix:"
Get-Content $OutputJson