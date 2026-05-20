# ─────────────────────────────────────────────────────────
# main.tf — Azure infrastructure for Sommelier
#
# Resources created:
#   - Resource Group
#   - Azure Container Registry (ACR)
#   - Azure Kubernetes Service (AKS) with cluster autoscaler
#   - Azure Storage Account + Container (for data files)
#   - Role assignments (AKS → ACR pull, AKS → Storage read)
# ─────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90"
    }
  }

  # Uncomment to store state in Azure Blob Storage (recommended for teams)
  # backend "azurerm" {
  #   resource_group_name  = "tfstate-rg"
  #   storage_account_name = "tfstatesommelier"
  #   container_name       = "tfstate"
  #   key                  = "sommelier.tfstate"
  # }
}

provider "azurerm" {
  features {}
}

# ── Resource Group ────────────────────────────────────────

resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location

  tags = var.tags
}

# ── Azure Container Registry ──────────────────────────────

resource "azurerm_container_registry" "acr" {
  name                = var.acr_name   # must be globally unique, alphanumeric only
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"        # use Standard/Premium for geo-replication
  admin_enabled       = false          # use managed identity instead of admin creds

  tags = var.tags
}

# ── AKS Cluster ───────────────────────────────────────────

resource "azurerm_kubernetes_cluster" "aks" {
  name                = var.aks_cluster_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = var.aks_cluster_name
  kubernetes_version  = var.kubernetes_version

  # System node pool — runs K8s system pods
  default_node_pool {
    name                = "system"
    node_count          = 1
    vm_size             = "Standard_D2s_v3"
    os_disk_size_gb     = 50
    type                = "VirtualMachineScaleSets"

    # Cluster autoscaler — scales nodes in/out based on pending pods
    enable_auto_scaling = true
    min_count           = 1
    max_count           = 3

    node_labels = {
      "nodepool-type" = "system"
    }
  }

  # Managed identity — no service principal to rotate
  identity {
    type = "SystemAssigned"
  }

  # Network config — required for ingress
  network_profile {
    network_plugin    = "azure"
    load_balancer_sku = "standard"
  }

  # Enable the metrics server (required for HPA)
  # It's enabled by default on AKS but explicit is clearer
  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  }

  tags = var.tags
}

# App node pool — runs your actual workloads
# Separate from system pool for isolation and independent scaling
resource "azurerm_kubernetes_cluster_node_pool" "app" {
  name                  = "app"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = var.app_node_vm_size
  os_disk_size_gb       = 100

  enable_auto_scaling = true
  min_count           = var.app_node_min_count
  max_count           = var.app_node_max_count

  node_labels = {
    "nodepool-type" = "app"
  }

  # Taint system pool so app pods only run here
  node_taints = []

  tags = var.tags
}

# ── ACR Pull Permission for AKS ───────────────────────────
# Allows AKS to pull images from ACR without credentials

resource "azurerm_role_assignment" "aks_acr_pull" {
  principal_id                     = azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id
  role_definition_name             = "AcrPull"
  scope                            = azurerm_container_registry.acr.id
  skip_service_principal_aad_check = true
}

# ── Storage Account for data files ───────────────────────
# Stores wines_clean.csv, wine_embeddings.npy, wine_faiss.index

resource "azurerm_storage_account" "data" {
  name                     = var.storage_account_name  # globally unique, lowercase
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  tags = var.tags
}

resource "azurerm_storage_container" "data" {
  name                  = "sommelier-data"
  storage_account_name  = azurerm_storage_account.data.name
  container_access_type = "private"
}

# ── Storage access for AKS ────────────────────────────────

resource "azurerm_role_assignment" "aks_storage_read" {
  principal_id         = azurerm_kubernetes_cluster.aks.identity[0].principal_id
  role_definition_name = "Storage Blob Data Reader"
  scope                = azurerm_storage_account.data.id
}

# ── Log Analytics (monitoring) ────────────────────────────

resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.aks_cluster_name}-logs"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = var.tags
}
