param(
    [string]$ProjectId,
    [string]$Region = "asia-southeast1",
    [string]$Zone = "asia-southeast1-a",
    [string]$ClusterName = "rl-placement-cluster",
    [int]$NodeCount = 5,
    [string]$MachineType = "e2-standard-2",
    [string]$ArtifactRepo = "rl-repo"
)

$ErrorActionPreference = "Stop"

if (-not $ProjectId) {
    throw "ProjectId is required. Example: .\scripts\provision_gke.ps1 -ProjectId your-project-id"
}

Write-Host "Setting project to $ProjectId"
gcloud config set project $ProjectId

Write-Host "Enabling required APIs"
gcloud services enable container.googleapis.com artifactregistry.googleapis.com compute.googleapis.com iam.googleapis.com storage.googleapis.com

Write-Host "Creating Artifact Registry repository if missing"
$repoExists = gcloud artifacts repositories list --location=$Region --project=$ProjectId --format="value(name)" |
    Where-Object { $_ -like "*/$ArtifactRepo" }
if (-not $repoExists) {
    gcloud artifacts repositories create $ArtifactRepo `
        --repository-format=docker `
        --location=$Region `
        --description="RL agent images" `
        --project=$ProjectId
}

Write-Host "Creating GKE cluster if missing"
$clusterExists = gcloud container clusters list --zone=$Zone --project=$ProjectId --format="value(name)" |
    Where-Object { $_ -eq $ClusterName }
if (-not $clusterExists) {
    gcloud container clusters create $ClusterName `
        --zone=$Zone `
        --num-nodes=$NodeCount `
        --machine-type=$MachineType `
        --disk-type=pd-standard `
        --disk-size=20 `
        --workload-pool="$ProjectId.svc.id.goog" `
        --project=$ProjectId
}

Write-Host "Fetching kubeconfig"
gcloud container clusters get-credentials $ClusterName --zone=$Zone --project=$ProjectId

Write-Host "Current nodes:"
kubectl get nodes -o wide
