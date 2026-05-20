variable "resource_group_name" {
  description = "Name of the Azure resource group"
  type        = string
  default     = "sommelier-rg"
}

variable "location" {
  description = "Azure region to deploy into"
  type        = string
  default     = "uksouth"   # change to eastus, westeurope, etc.
}

variable "acr_name" {
  description = "Azure Container Registry name — must be globally unique, alphanumeric only, 5-50 chars"
  type        = string
  # Example: "sommelieracr123"
}

variable "aks_cluster_name" {
  description = "Name of the AKS cluster"
  type        = string
  default     = "sommelier-aks"
}

variable "kubernetes_version" {
  description = "Kubernetes version for AKS"
  type        = string
  default     = "1.29"
}

variable "app_node_vm_size" {
  description = "VM size for app node pool. D2s_v3 = 2 vCPU, 8GB RAM"
  type        = string
  default     = "Standard_D2s_v3"
}

variable "app_node_min_count" {
  description = "Minimum nodes in app pool (for autoscaler)"
  type        = number
  default     = 1
}

variable "app_node_max_count" {
  description = "Maximum nodes in app pool (for autoscaler)"
  type        = number
  default     = 5
}

variable "storage_account_name" {
  description = "Storage account name — globally unique, lowercase, 3-24 chars"
  type        = string
  # Example: "sommelierdatastore"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    project     = "sommelier"
    environment = "production"
    managed-by  = "terraform"
  }
}
