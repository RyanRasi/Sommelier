# Sommelier — AI Wine Recommendation System

A natural language wine recommendation system powered by semantic search and a local LLM. Describe what you want in plain English and get back curated, explained recommendations from a dataset of 120,000 wines.

---

## Overview

Type something like *"bold red for a medium-rare steak"* or *"dry French wine under $30"* and the system:

1. Uses an LLM to extract structured filters (country, price, variety, flavour profile)
2. Applies those filters to narrow a 120k wine dataset
3. Runs semantic search using sentence embeddings + FAISS vector search
4. Passes the top candidates back to the LLM to select and explain the best 3

Everything runs locally — no OpenAI API key required.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  LLM Filter         │  Extracts: country, max_price, variety,
│  Extraction         │  flavour keywords, cleaned search query
│  (Ollama / llama3.2)│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Hard Filter        │  Narrows 120k wines to matching subset
│  (pandas)           │  by country, price, variety, points
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Semantic Search    │  Embeds cleaned query → searches FAISS
│  (FAISS + sentence- │  index → returns top 15 candidates
│   transformers)     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LLM Refinement     │  Selects best 3, writes explanations,
│  (Ollama / llama3.2)│  food pairings, and serving tips
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  FastAPI            │  REST API served on localhost:8000
│  REST API           │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  React Frontend     │  Search UI with animated cards
│  (Vite)             │  served on localhost:5173
└─────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384 dimensions) |
| Vector search | `FAISS` — exact cosine similarity search |
| LLM | `Ollama` running `llama3.2` locally |
| API | `FastAPI` + `uvicorn` |
| Frontend | `React 19` + `Vite` |
| Dataset | Wine Reviews — 120k wines |
| Containers | Docker |
| Orchestration | Kubernetes (AKS) |
| Infrastructure | Terraform |
| Cloud | Azure (AKS, ACR, Blob Storage) |

---

## Project Structure

```
.
├── api/
│   ├── api.py                    # FastAPI REST API
│   ├── recommender.py            # Full recommendation pipeline
│   └── test_api.py               # API integration tests
├── ingest/
│   ├── main.py                   # Orchestrates data pipeline
│   ├── clean_and_preprocess.py   # Data cleaning
│   ├── generate_embeddings.py    # Sentence embeddings
│   ├── faiss_index.py            # FAISS index builder
│   └── load.py                   # Data loading utilities
├── sommelier-ui/                 # React frontend (Vite)
│   ├── src/
│   │   ├── App.jsx               # Main UI component
│   │   └── main.jsx              # React entry point
│   ├── package.json
│   └── vite.config.js
├── infrastructure/
│   ├── k8s/                      # Kubernetes manifests
│   │   ├── namespace.yaml
│   │   ├── api-deployment.yaml
│   │   ├── frontend-deployment.yaml
│   │   ├── ollama-deployment.yaml
│   │   ├── ingress.yaml
│   │   ├── hpa.yaml
│   │   └── secrets.yaml
│   ├── terraform/                # Azure infrastructure as code
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── terraform.tfvars.example
│   ├── scripts/
│   │   ├── build-push.sh         # Build & push Docker images to ACR
│   │   └── upload-data.sh        # Upload data files to Blob Storage
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   └── nginx.conf
├── requirements.txt
├── wines_clean.csv               # Cleaned wine dataset
├── wine_embeddings.npy           # Pre-computed embeddings (~180MB)
└── wine_faiss.index              # FAISS vector index
```

---

# Running Locally

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Backend runtime |
| Node.js | 18+ | Frontend build |
| Ollama | Latest | Local LLM |

Verify your installs:

```bash
python --version   # 3.9+
node --version     # v18+
ollama --version
```

---

## Step 1 — Set up Python environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

---

## Step 2 — Data files

The data file `wine-reviews.zip`, must be downloaded from Kaggle and placed in ingest/wine-data/.

Then to generate the embeddings and index from the raw dataset:

```bash
cd ingest
python main.py
```

> This takes 10–20 minutes on first run while generating embeddings.

---

## Step 3 — Pull the LLM model

```bash
ollama pull llama3.2
```

Downloads `llama3.2` (~2GB) once to your local Ollama store. Verify with:

```bash
ollama list   # should show llama3.2
```

---

## Step 4 — Configure the frontend API URL

Open `sommelier-ui/src/App.jsx` and set `API_URL` to your local machine's IP or `localhost`:

```javascript
const API_URL = "http://localhost:8000";
```

---

## Step 5 — Install frontend dependencies

```bash
cd sommelier-ui
npm install
cd ..
```

---

## Step 6 — Run the app

You need two terminals.

**Terminal 1 — FastAPI backend:**

```bash
uvicorn api.api:app --reload --port 8000
```

Expected:
```
INFO: Uvicorn running on http://127.0.0.1:8000
INFO: 119895 wines loaded and ready.
```

**Terminal 2 — React frontend:**

```bash
cd sommelier-ui
npm run dev
```

Expected:
```
VITE v8.x.x  ready in ~300ms
Local: http://localhost:5173/
```

Open **http://localhost:5173** in your browser.

---

## Verification

| URL | Expected |
|---|---|
| `http://localhost:5173` | React UI |
| `http://localhost:8000/health` | `{"status": "healthy", ...}` |
| `http://localhost:8000/docs` | Swagger UI |

---

# Deploying to Azure

The infrastructure deploys to Azure Kubernetes Service (AKS) with:

- **ACR** — Azure Container Registry for Docker images
- **AKS** — Kubernetes cluster with autoscaling
- **Azure Blob Storage** — hosts the large data files
- **Log Analytics** — monitoring

## Prerequisites

Install:
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
- [Terraform](https://developer.hashicorp.com/terraform/install) 1.5.0+
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Helm](https://helm.sh/docs/intro/install/)
- [Docker](https://docs.docker.com/get-docker/)

Authenticate to Azure:

```bash
az login
az account set --subscription "<your-subscription-id>"
```

---

## Step 1 — Configure Terraform variables

```bash
cd infrastructure/terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` — the ACR name and storage account name must be globally unique:

```hcl
resource_group_name  = "sommelier-rg"
location             = "uksouth"
acr_name             = "sommelieracr"        # must be globally unique, lowercase, no hyphens
aks_cluster_name     = "sommelier-aks"
storage_account_name = "sommelierdatastore"  # must be globally unique, lowercase, no hyphens
app_node_min_count   = 1
app_node_max_count   = 5
app_node_vm_size     = "Standard_D2s_v3"
kubernetes_version   = "1.29"
```

---

## Step 2 — Provision Azure infrastructure

```bash
terraform init
terraform plan
terraform apply
```

This creates the resource group, ACR, AKS cluster, storage account, and Log Analytics workspace. It takes ~10 minutes.

Note the outputs — you'll need the ACR login server and storage account name in later steps.

---

## Step 3 — Connect kubectl to AKS

```bash
az aks get-credentials \
  --resource-group sommelier-rg \
  --name sommelier-aks
```

Verify:

```bash
kubectl get nodes
```

---

## Step 4 — Create the Kubernetes namespace and secrets

```bash
cd infrastructure
kubectl apply -f k8s/namespace.yaml

kubectl create secret generic sommelier-secrets \
  --namespace sommelier \
  --from-literal=storage-account-name=<your-storage-account-name>
```

---

## Step 5 — Upload data files to Blob Storage

The large data files are downloaded by the API pod's init container at startup. Upload them first:

```bash
./scripts/upload-data.sh <your-storage-account-name>
```

This uploads `wines_clean.csv`, `wine_embeddings.npy`, and `wine_faiss.index` to a Blob Storage container named `sommelier-data`.

---

## Step 6 — Build and push Docker images

```bash
./scripts/build-push.sh <your-acr-name>
```

This:
1. Logs into ACR
2. Builds `Dockerfile.api` and `Dockerfile.frontend`
3. Pushes both images with a timestamp tag and `latest`
4. Updates the K8s deployment YAMLs with your ACR name
5. Restarts deployments to pull the new images

---

## Step 7 — Install the NGINX ingress controller

```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace
```

---

## Step 8 — Deploy to Kubernetes

```bash
kubectl apply -f k8s/
```

Watch pods come up:

```bash
kubectl get pods -n sommelier -w
```

The Ollama pod's init container will pull the `llama3.2` model (~2GB) on first start — this takes a few minutes.

---

## Step 9 — Get the public IP

```bash
kubectl get service -n ingress-nginx ingress-nginx-controller
```

The `EXTERNAL-IP` column is your app's public URL. It may show `<pending>` for a minute while Azure provisions the load balancer.

---

## Azure Resource Overview

| Resource | Purpose | Size |
|---|---|---|
| AKS — system node pool | Cluster control plane workloads | 1 node (autoscales 1–3) |
| AKS — app node pool | API, frontend, Ollama pods | `Standard_D2s_v3`, autoscales 1–5 |
| ACR | Docker image storage | — |
| Blob Storage | Wine data files | ~400MB |
| Log Analytics | Monitoring | 30-day retention |

### Pod resource limits

| Pod | CPU Request | CPU Limit | Memory Request | Memory Limit |
|---|---|---|---|---|
| API | 500m | 2 vCPU | 1Gi | 4Gi |
| Frontend | 100m | 500m | 128Mi | 256Mi |
| Ollama | 1 vCPU | 4 vCPU | 4Gi | 8Gi |

### Autoscaling (HPA)

| Deployment | Min Replicas | Max Replicas | Scale trigger |
|---|---|---|---|
| API | 2 | 10 | 70% CPU / 80% memory |
| Frontend | 2 | 8 | 70% CPU |

---

## Enabling HTTPS

The ingress manifest (`k8s/ingress.yaml`) includes commented-out cert-manager annotations. To enable TLS:

1. Install cert-manager: `helm install cert-manager jetstack/cert-manager --set installCRDs=true`
2. Uncomment the TLS block in `ingress.yaml` and set your domain
3. Re-apply: `kubectl apply -f k8s/ingress.yaml`

---

# API Reference

Base URL: `http://localhost:8000` (local) or your ingress IP (Azure)

### `GET /`

Returns API status.

```json
{
  "name": "AI Sommelier API",
  "status": "running",
  "wines_loaded": 119895,
  "usage": "POST /recommend with {query: string}",
  "docs": "/docs"
}
```

### `GET /health`

Health check used by Kubernetes readiness and liveness probes.

```json
{
  "status": "healthy",
  "wines_loaded": 119895,
  "model": "llama3.2",
  "version": "1.0.0"
}
```

### `POST /recommend`

Returns 3 wine recommendations for a natural language query.

**Request:**
```json
{
  "query": "dry wine from France under $30"
}
```

**Response `200`:**
```json
{
  "query": "dry wine from France under $30",
  "recommendations": [
    {
      "rank": 1,
      "title": "Château Pichon Baron 2012 (Pauillac)",
      "why": "This classic Bordeaux is the definition of dry and structured...",
      "food_pairing": "Perfect with lamb chops or duck confit.",
      "serving_tip": "Decant for 30 minutes before serving."
    }
  ],
  "count": 3,
  "elapsed_seconds": 4.21
}
```

**Errors:**

| Status | Reason |
|---|---|
| `400` | Empty query or query over 500 characters |
| `404` | No recommendations found — try rephrasing |
| `500` | Pipeline error |

Interactive Swagger docs: `http://localhost:8000/docs`

---

# Example Queries

```
wine for medium rare steak
fruity and light wine
dry wine from France
Italian red under $25
highly rated wine over 95 points
something sweet for dessert
bold earthy Barolo
cheap everyday white wine
wine to pair with salmon
```

---

## License

MIT
