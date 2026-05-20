#!/bin/bash
# ─────────────────────────────────────────────────────────
# upload-data.sh
#
# Uploads the large data files to Azure Blob Storage.
# Run this once after generating the embeddings locally.
# AKS pods will download these files on startup via the
# init container defined in api-deployment.yaml.
#
# Usage:
#   ./scripts/upload-data.sh <storage-account-name>
#
# Example:
#   ./scripts/upload-data.sh sommelierdatastore
# ─────────────────────────────────────────────────────────

set -euo pipefail

STORAGE_ACCOUNT="${1:-}"
CONTAINER="sommelier-data"

if [ -z "$STORAGE_ACCOUNT" ]; then
  echo "Usage: $0 <storage-account-name>"
  echo "Example: $0 sommelierdatastore"
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Uploading data files to Azure Blob Storage"
echo "  Storage account: $STORAGE_ACCOUNT"
echo "  Container: $CONTAINER"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check all files exist before starting
FILES=(
  "wines_clean.csv"
  "wine_embeddings.npy"
  "wine_faiss.index"
)

echo ""
echo "▶ Checking files exist..."
for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "  ❌ Missing: $f"
    echo "  Run your preprocessing pipeline first to generate this file."
    exit 1
  fi
  SIZE=$(du -sh "$f" | cut -f1)
  echo "  ✅ $f ($SIZE)"
done

# Upload each file
echo ""
echo "▶ Uploading files (this may take a few minutes)..."

for f in "${FILES[@]}"; do
  echo ""
  echo "  Uploading $f..."
  az storage blob upload \
    --account-name "$STORAGE_ACCOUNT" \
    --container-name "$CONTAINER" \
    --name "$f" \
    --file "$f" \
    --auth-mode login \
    --overwrite

  echo "  ✅ $f uploaded."
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ All files uploaded successfully."
echo ""
echo "  Verify with:"
echo "  az storage blob list \\"
echo "    --account-name $STORAGE_ACCOUNT \\"
echo "    --container-name $CONTAINER \\"
echo "    --auth-mode login \\"
echo "    --output table"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
