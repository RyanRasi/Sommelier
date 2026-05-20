# Copy this file to terraform.tfvars and fill in your values
# terraform.tfvars is in .gitignore — never commit real values

resource_group_name  = "sommelier-rg"
location             = "uksouth"
acr_name             = "sommelieracr"        # must be globally unique
aks_cluster_name     = "sommelier-aks"
storage_account_name = "sommelierdatastore"  # must be globally unique

# Scale up if you need more capacity
app_node_min_count = 1
app_node_max_count = 5
app_node_vm_size   = "Standard_D2s_v3"
