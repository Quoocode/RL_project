param(
    [string]$ProjectId,
    [string]$Region = "asia-southeast1",
    [string]$Zone = "asia-southeast1-a",
    [string]$ClusterName = "rl-placement-cluster",
    [string]$ArtifactRepo = "rl-repo"
)

$ErrorActionPreference = "Stop"

if (-not $ProjectId) {
    throw "ProjectId is required. Example: .\provision_gke.ps1 -ProjectId k8s-rl-project"
}

function Check-LastCommand {
    param(
        [string]$Message
    )

    if ($LASTEXITCODE -ne 0) {
        throw $Message
    }
}

Write-Host "Setting project to $ProjectId"
gcloud config set project $ProjectId
Check-LastCommand "Failed to set project"

Write-Host "Setting ADC quota project to $ProjectId"
gcloud auth application-default set-quota-project $ProjectId
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Could not set ADC quota project. Continue..."
}

Write-Host "Enabling required APIs"
gcloud services enable `
    container.googleapis.com `
    artifactregistry.googleapis.com `
    compute.googleapis.com `
    iam.googleapis.com `
    storage.googleapis.com `
    --project=$ProjectId
Check-LastCommand "Failed to enable required APIs"

# ------------------------------------------------------------
# Artifact Registry
# ------------------------------------------------------------
Write-Host "Checking Artifact Registry repository: $ArtifactRepo"

$repoName = gcloud artifacts repositories list `
    --location=$Region `
    --project=$ProjectId `
    --filter="name:$ArtifactRepo" `
    --format="value(name)"

if (-not $repoName) {
    Write-Host "Creating Artifact Registry repository: $ArtifactRepo"

    gcloud artifacts repositories create $ArtifactRepo `
        --repository-format=docker `
        --location=$Region `
        --description="RL agent images" `
        --project=$ProjectId

    Check-LastCommand "Failed to create Artifact Registry repository"
}
else {
    Write-Host "Artifact Registry repository already exists: $ArtifactRepo"
}

# ------------------------------------------------------------
# GKE Cluster
# ------------------------------------------------------------
Write-Host "Checking existing GKE cluster: $ClusterName"

$clusterNameFound = gcloud container clusters list `
    --zone=$Zone `
    --project=$ProjectId `
    --filter="name=$ClusterName" `
    --format="value(name)"

if ($clusterNameFound -eq $ClusterName) {
    Write-Host "Cluster already exists: $ClusterName"
    Write-Host "If this is the old cluster, delete it first:"
    Write-Host "gcloud container clusters delete $ClusterName --zone=$Zone --project=$ProjectId"
}
else {
    Write-Host "Creating GKE cluster with heterogeneous node pools"
    Write-Host "small/default pool: 2 x e2-standard-2"

    gcloud container clusters create $ClusterName `
        --zone=$Zone `
        --num-nodes=2 `
        --machine-type=e2-standard-2 `
        --disk-type=pd-standard `
        --disk-size=20 `
        --workload-pool="$ProjectId.svc.id.goog" `
        "--node-labels=node-size=small,rl-node-index=small" `
        --project=$ProjectId

    Check-LastCommand "Failed to create GKE cluster"

    Write-Host "Creating medium node pool"
    Write-Host "medium-pool: 2 x e2-standard-4"

    gcloud container node-pools create medium-pool `
        --cluster=$ClusterName `
        --zone=$Zone `
        --num-nodes=2 `
        --machine-type=e2-standard-4 `
        --disk-type=pd-standard `
        --disk-size=20 `
        "--node-labels=node-size=medium,rl-node-index=medium" `
        --project=$ProjectId

    Check-LastCommand "Failed to create medium node pool"

    Write-Host "Creating large node pool"
    Write-Host "large-pool: 1 x e2-standard-8"

    gcloud container node-pools create large-pool `
        --cluster=$ClusterName `
        --zone=$Zone `
        --num-nodes=1 `
        --machine-type=e2-standard-8 `
        --disk-type=pd-standard `
        --disk-size=20 `
        "--node-labels=node-size=large,rl-node-index=large" `
        --project=$ProjectId

    Check-LastCommand "Failed to create large node pool"
}

# ------------------------------------------------------------
# Kubeconfig
# ------------------------------------------------------------
Write-Host "Fetching kubeconfig"

gcloud container clusters get-credentials $ClusterName `
    --zone=$Zone `
    --project=$ProjectId

Check-LastCommand "Failed to fetch kubeconfig"

# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------
Write-Host ""
Write-Host "Current Kubernetes nodes:"
kubectl get nodes -o wide

Write-Host ""
Write-Host "Node labels:"
kubectl get nodes --show-labels

Write-Host ""
Write-Host "Kubernetes allocatable resources:"
kubectl get nodes -o custom-columns="NAME:.metadata.name,CPU:.status.allocatable.cpu,MEM:.status.allocatable.memory"

Write-Host ""
Write-Host "GCE instance machine types:"
gcloud compute instances list `
    --filter="name~'gke-$ClusterName'" `
    --format="table(name,zone,machineType.basename())" `
    --project=$ProjectId

Write-Host ""
Write-Host "Done."
Write-Host "Expected node configuration:"
Write-Host "  2 x e2-standard-2"
Write-Host "  2 x e2-standard-4"
Write-Host "  1 x e2-standard-8"

# ------------------------------------------------------------
# Export cluster_state.json for downstream scripts
# ------------------------------------------------------------
Write-Host ""
Write-Host "Exporting cluster_state.json"

$Namespace = "drl-scheduler"

kubectl create namespace $Namespace 2>$null

function Get-NodeByLabel {
    param(
        [string]$Size
    )

    $nodesRaw = kubectl get nodes `
        -l "node-size=$Size" `
        -o jsonpath="{range .items[*]}{.metadata.name}{`n`}{end}"

    return @(
        $nodesRaw -split "`n" |
        Where-Object { $_.Trim() -ne "" } |
        Sort-Object
    )
}

$smallNodes = Get-NodeByLabel "small"
$mediumNodes = Get-NodeByLabel "medium"
$largeNodes = Get-NodeByLabel "large"

if ($smallNodes.Count -lt 2) {
    throw "Need at least 2 small nodes, found $($smallNodes.Count)"
}

if ($mediumNodes.Count -lt 2) {
    throw "Need at least 2 medium nodes, found $($mediumNodes.Count)"
}

if ($largeNodes.Count -lt 1) {
    throw "Need at least 1 large node, found $($largeNodes.Count)"
}

$clusterState = [ordered]@{
    cluster_name = $ClusterName
    namespace = $Namespace
    node_order = @(
        [ordered]@{
            index = 0
            role = "small"
            name = $smallNodes[0]
        },
        [ordered]@{
            index = 1
            role = "medium"
            name = $mediumNodes[0]
        },
        [ordered]@{
            index = 2
            role = "large"
            name = $largeNodes[0]
        },
        [ordered]@{
            index = 3
            role = "medium"
            name = $mediumNodes[1]
        },
        [ordered]@{
            index = 4
            role = "small"
            name = $smallNodes[1]
        }
    )
}

$clusterState | ConvertTo-Json -Depth 10 | Out-File -Encoding utf8 "cluster_state.json"

Write-Host "Saved cluster_state.json"
Get-Content "cluster_state.json"