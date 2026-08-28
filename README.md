# Financial Document Processing System

Agentic harness for extracting, validating, and auditing financial documents.

## How It Works:

1. **Supervisor Agent** receives a document
2. `check_authority` → Validates GSTIN, business ID/name or other income tax department related stuff by calling the government API(s). Gated with deterministic checksum and modular arithmetic on the IDs.
2. `delegate_extraction` → Parses the document by running a gemini & sarvam instance in parallel. Final extraction is mathematically gated (e.g. column sum matching extracted total) and retried automatically .
3. `execute_code` → Stateful python environment for zooming or cropping images to verify the extraction with more accuracy. Gemini extractor agent gets this tool too.
4. `check_fraud_vectors` → Harness goes through the weviate for checking duplicates and traverse the nebulagraph to check the relationship between extracted entities. Finds other entities in the vicinity of the extracted entities.
5. `inspect_past_records` → Inspect past records by providing ID. Goes through nebula graph and weviate again to provide useful necessary information.
6. Calls `update_records` → Saves to database OR `ignore_request` → Rejects

## Architecture

- LangGraph ( Supervisor + Extractor ) + Weaviate (Vector DB) + NebulaGraph (Graph DB)
- **Weaviate** (Vector DB): Similarity search & fraud detection via embeddings
- **NebulaGraph** (Graph DB): Entity relationships tracking
- **FastAPI Backend**: Serves the agents via REST & WebSocket APIs
- **React Frontend**: Professional Agentic UI for visualizing agent activity


## Setup

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 2. Start Databases

**Both databases are required for full functionality.**

```bash
docker compose -f backend/docker-compose-weaviate.yml up -d
docker compose -f backend/docker-compose-nebula.yml up -d
```
(use `down` to stop the databases)

#### Verify Databases are Running
```bash
docker ps
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "weaviate|nebula|financial"
```

### 3. Run the System

#### Option A: Web UI (Recommended)

**Start Backend:**
```bash
./venv/bin/uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
```

**Start Frontend (in a new terminal):**
```bash
cd frontend
npm run dev
```

Then open your browser to `http://localhost:5173`

#### Option B: CLI

```bash
python3 -m backend.full_agent_test --input your_receipt.jpg
```