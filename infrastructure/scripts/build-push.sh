#!/bin/bash
# ─────────────────────────────────────────────────────────
# build-push.sh
#
# Builds Docker images and pushes them to Azure Container
# Registry, then updates the K8s deployments to use the
# new images.
#
# Usage:
#   ./scripts/build-push.sh <acr-name>
#
# Example:
#   ./scripts/build-push.sh sommelieracr
# ─────────────────────────────────────────────────────────

set -euo pipefail

ACR_NAME="${1:-}"

if [ -z "$ACR_NAME" ]; then
  echo "Usage: $0 <acr-name>"
  echo "Example: $0 sommelieracr"
  exit 1
fi

ACR_SERVER="${ACR_NAME}.azurecr.io"
TIMESTAMP=$(date +%Y%m%d%H%M%S)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Building and pushing to: $ACR_SERVER"
echo "  Timestamp tag: $TIMESTAMP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Log in to ACR ──────────────────────────────────────
echo ""
echo "▶ Logging in to ACR..."
az acr login --name "$ACR_NAME"

# ── 2. Build and push API image ───────────────────────────
echo ""
echo "▶ Building API image..."
docker build \
  -f Dockerfile.api \
  -t "${ACR_SERVER}/sommelier-api:${TIMESTAMP}" \
  -t "${ACR_SERVER}/sommelier-api:latest" \
  .

echo "▶ Pushing API image..."
docker push "${ACR_SERVER}/sommelier-api:${TIMESTAMP}"
docker push "${ACR_SERVER}/sommelier-api:latest"

# ── 3. Build and push frontend image ─────────────────────
echo ""
echo "▶ Building frontend image..."
docker build \
  -f Dockerfile.frontend \
  -t "${ACR_SERVER}/sommelier-frontend:${TIMESTAMP}" \
  -t "${ACR_SERVER}/sommelier-frontend:latest" \
  .

echo "▶ Pushing frontend image..."
docker push "${ACR_SERVER}/sommelier-frontend:${TIMESTAMP}"
docker push "${ACR_SERVER}/sommelier-frontend:latest"

# ── 4. Update image references in K8s manifests ──────────
echo ""
echo "▶ Updating K8s manifests with ACR name..."
sed -i "s|REPLACE_WITH_ACR_NAME|${ACR_NAME}|g" k8s/api-deployment.yaml
sed -i "s|REPLACE_WITH_ACR_NAME|${ACR_NAME}|g" k8s/frontend-deployment.yaml

# ── 5. Restart deployments to pull latest images ─────────
echo ""
echo "▶ Rolling out new images to AKS..."
kubectl rollout restart deployment/api      -n sommelier
kubectl rollout restart deployment/frontend -n sommelier

echo ""
echo "▶ Waiting for rollouts to complete..."
kubectl rollout status deployment/api      -n sommelier --timeout=180s
kubectl rollout status deployment/frontend -n sommelier --timeout=180s

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Deploy complete!"
echo ""
echo "  API image:      ${ACR_SERVER}/sommelier-api:${TIMESTAMP}"
echo "  Frontend image: ${ACR_SERVER}/sommelier-frontend:${TIMESTAMP}"
echo ""
echo "  Get public IP:"
echo "  kubectl get service -n ingress-nginx ingress-nginx-controller"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
