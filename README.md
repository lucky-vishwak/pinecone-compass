# Pinecone Compass

A MongoDB Compass-style browser for **Pinecone** indexes.

- Left sidebar lists every Pinecone index (expand to see namespaces + record counts).
- Click a namespace → loads with the default filter `{}` at 10 records per page.
- Type any Pinecone **metadata filter** as JSON in the Filter box (e.g. `{ "tenantId": "322" }`) and hit **Find** / Enter.
- Records render as Compass-style syntax-highlighted documents with paging.

Credentials are read from this repo's **own** env file — `.env` if present,
otherwise `.env.dev` (committed with the Pinecone keys). Nothing outside this
folder is needed.

## Run

```bash
cd replica_repos/pinecone-compass
npm install
npm start
```

Open http://localhost:6070

## Config (env)

Set in `.env` (preferred) or `.env.dev` in this folder:

| Var | Default | Purpose |
|-----|---------|---------|
| `PINECONE_API` | — | Pinecone API key (required) |
| `PINECONE_CLOUD` | — | Pinecone cloud (e.g. `aws`) |
| `PINECONE_REGION` | — | Pinecone region (e.g. `us-east-1`) |
| `PC_UI_PORT` | `6070` | Port to serve the UI |

## How browsing works

Pinecone has no "scan all rows" API, so the server queries each index against a
fixed constant dummy vector (`includeMetadata: true`) and slices the matches into
pages. Metadata filters are passed straight through to Pinecone's `filter`. For an
empty filter the total comes from `describeIndexStats`; for a metadata filter it is
the matched-record count (capped at Pinecone's `topK` limit of 10000).
