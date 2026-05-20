output "acr_login_server" {
  description = "ACR login server URL — use this to tag and push images"
  value       = azurerm_container_registry.acr.login_server
}

output "aks_get_credentials_command" {
  description = "Run this to configure kubectl to point at your AKS cluster"
  value       = "az aks get-credentials --resource-group ${azurerm_resource_group.main.name} --name ${azurerm_kubernetes_cluster.aks.name}"
}

output "storage_account_name" {
  description = "Storage account name — used by upload-data.sh"
  value       = azurerm_storage_account.data.name
}

output "storage_container_name" {
  description = "Blob container name — used by upload-data.sh"
  value       = azurerm_storage_container.data.name
}

output "resource_group_name" {
  description = "Resource group name"
  value       = azurerm_resource_group.main.name
}
