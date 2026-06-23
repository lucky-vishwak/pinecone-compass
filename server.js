"use strict";

const fs = require("fs");
const path = require("path");
const express = require("express");
const { Pinecone } = require("@pinecone-database/pinecone");

// Load Pinecone credentials from this repo's own env file: .env, else .env.dev.
const LOCAL_ENV = path.resolve(__dirname, ".env");
const LOCAL_ENV_DEV = path.resolve(__dirname, ".env.dev");
const ENV_PATH = fs.existsSync(LOCAL_ENV) ? LOCAL_ENV : LOCAL_ENV_DEV;
require("dotenv").config({ path: ENV_PATH });

// Hosting providers (Render, Railway, etc.) inject PORT; fall back for local.
const PORT = process.env.PORT || process.env.PC_UI_PORT || 6070;
const MAX_TOPK = 10000; // Pinecone hard cap for query topK
const DEFAULT_NAMESPACE = "__default__";

if (!process.env.PINECONE_API) {
    // eslint-disable-next-line no-console
    console.error(
        `PINECONE_API not found. Looked in env file: ${ENV_PATH}`
    );
    process.exit(1);
}

const pc = new Pinecone({ apiKey: process.env.PINECONE_API });

// Cache index dimension/metric so we can build a dummy browse vector.
const indexMetaCache = new Map();

async function getIndexMeta(name) {
    if (indexMetaCache.has(name)) return indexMetaCache.get(name);
    const list = await pc.listIndexes();
    (list.indexes || []).forEach((i) => indexMetaCache.set(i.name, i));
    return indexMetaCache.get(name);
}

// Pinecone has no "scan" API, so to browse records we run a similarity query
// against a constant non-zero vector (a zero vector returns nothing on cosine
// indexes). Results are deterministic for a fixed dummy vector, which lets us
// slice them into pages.
function dummyVector(dim) {
    return new Array(dim).fill(0.1);
}

// Compass-style projection: { field: 1 } include-only, { field: 0 } exclude.
// Mixing is treated as include-mode (any 1 wins), matching Mongo's behaviour.
function applyProject(metadata, project) {
    if (!project || typeof project !== "object") return metadata;
    const keys = Object.keys(project);
    if (!keys.length) return metadata;
    const includeMode = keys.some((k) => Number(project[k]) === 1);
    const out = {};
    if (includeMode) {
        keys.forEach((k) => {
            if (Number(project[k]) === 1 && k in metadata) out[k] = metadata[k];
        });
    } else {
        Object.keys(metadata).forEach((k) => {
            if (Number(project[k]) !== 0) out[k] = metadata[k];
        });
    }
    return out;
}

const app = express();

// HTTP Basic Auth gate. Enabled whenever APP_PASSWORD is set (always set it in
// production). Username defaults to "admin" if APP_USER is not provided.
const AUTH_USER = process.env.APP_USER || "admin";
const AUTH_PASS = process.env.APP_PASSWORD || "";
if (AUTH_PASS) {
    app.use((req, res, next) => {
        const header = req.headers.authorization || "";
        const [scheme, encoded] = header.split(" ");
        if (scheme === "Basic" && encoded) {
            const [user, pass] = Buffer.from(encoded, "base64")
                .toString()
                .split(":");
            if (user === AUTH_USER && pass === AUTH_PASS) return next();
        }
        res.set("WWW-Authenticate", 'Basic realm="Pinecone Compass"');
        return res.status(401).send("Authentication required.");
    });
} else {
    // eslint-disable-next-line no-console
    console.warn(
        "WARNING: APP_PASSWORD is not set — the app is UNPROTECTED. " +
            "Set APP_PASSWORD before exposing it publicly."
    );
}

app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

// List all indexes with their namespaces + record counts (the left sidebar).
app.get("/api/indexes", async (req, res) => {
    try {
        const list = await pc.listIndexes();
        const indexes = list.indexes || [];
        indexes.forEach((i) => indexMetaCache.set(i.name, i));

        const result = await Promise.all(
            indexes.map(async (idx) => {
                let namespaces = [];
                let total = 0;
                try {
                    const stats = await pc
                        .index(idx.name)
                        .describeIndexStats();
                    total = stats.totalRecordCount || 0;
                    namespaces = Object.entries(stats.namespaces || {}).map(
                        ([name, v]) => ({
                            name,
                            count: v.recordCount || 0
                        })
                    );
                } catch (e) {
                    // index may be initializing; surface it but don't fail all
                }
                return {
                    name: idx.name,
                    dimension: idx.dimension,
                    metric: idx.metric,
                    host: idx.host,
                    total,
                    namespaces
                };
            })
        );
        res.json({ status: "success", data: result });
    } catch (err) {
        res.status(500).json({ status: "error", message: err.message });
    }
});

// Query/browse records for an index with an optional metadata filter + paging.
app.post("/api/query", async (req, res) => {
    try {
        const {
            index: indexName,
            namespace = "",
            filter = {},
            project = {},
            sort = {},
            limit = 0,
            page = 1,
            pageSize = 10
        } = req.body || {};

        if (!indexName) {
            return res
                .status(400)
                .json({ status: "error", message: "index is required" });
        }

        const meta = await getIndexMeta(indexName);
        if (!meta) {
            return res
                .status(404)
                .json({ status: "error", message: "index not found" });
        }

        const pg = Math.max(1, parseInt(page, 10) || 1);
        const size = Math.max(1, Math.min(1000, parseInt(pageSize, 10) || 10));
        const ns =
            namespace && namespace !== DEFAULT_NAMESPACE ? namespace : "";

        const hasFilter = filter && Object.keys(filter).length > 0;
        const sortKeys = sort && typeof sort === "object" ? Object.keys(sort) : [];
        const hasSort = sortKeys.length > 0;
        const lim = Math.max(0, parseInt(limit, 10) || 0);

        // Pinecone can't sort or project, so when a Sort/Limit is set we pull a
        // larger candidate set and order/trim it server-side, then page over it.
        let need;
        if (lim > 0) {
            need = Math.min(MAX_TOPK, lim);
        } else if (hasSort) {
            need = MAX_TOPK;
        } else {
            need = Math.min(MAX_TOPK, pg * size);
        }
        const target = ns ? pc.index(indexName).namespace(ns) : pc.index(indexName);

        const queryParams = {
            vector: dummyVector(meta.dimension),
            topK: need,
            includeMetadata: true,
            includeValues: false
        };
        if (hasFilter) queryParams.filter = filter;

        const result = await target.query(queryParams);
        let matches = result.matches || [];

        // Sort: { field: 1 } asc, { field: -1 } desc — compared on metadata.
        if (hasSort) {
            matches = matches.slice().sort((a, b) => {
                for (const k of sortKeys) {
                    const dir = Number(sort[k]) < 0 ? -1 : 1;
                    const av = (a.metadata || {})[k];
                    const bv = (b.metadata || {})[k];
                    if (av === bv) continue;
                    if (av === undefined || av === null) return 1;
                    if (bv === undefined || bv === null) return -1;
                    if (av < bv) return -1 * dir;
                    if (av > bv) return 1 * dir;
                }
                return 0;
            });
        }

        // Limit caps the total result set (applied after sort, before paging).
        if (lim > 0) matches = matches.slice(0, lim);

        const start = (pg - 1) * size;
        const rows = matches.slice(start, start + size).map((m) => ({
            id: m.id,
            score: m.score,
            metadata: applyProject(m.metadata || {}, project)
        }));

        // Total: for an unfiltered, unsorted, unlimited browse use stats;
        // otherwise the matched count (capped at MAX_TOPK like Compass tools).
        let total = matches.length;
        let totalCapped = !hasFilter ? false : matches.length >= MAX_TOPK;
        if (!hasFilter && !hasSort && lim === 0) {
            try {
                const stats = await target.describeIndexStats();
                if (ns) {
                    total =
                        (stats.namespaces &&
                            stats.namespaces[ns] &&
                            stats.namespaces[ns].recordCount) ||
                        matches.length;
                } else {
                    total = stats.totalRecordCount || matches.length;
                }
                totalCapped = false;
            } catch (e) {
                /* keep matches.length */
            }
        }

        res.json({
            status: "success",
            data: {
                rows,
                page: pg,
                pageSize: size,
                total,
                totalCapped,
                returned: rows.length
            }
        });
    } catch (err) {
        res.status(500).json({ status: "error", message: err.message });
    }
});

// Export every record matching the current filter (+ sort/limit/project) as a
// downloadable JSON file. Capped at MAX_TOPK like the rest of the app.
app.post("/api/export", async (req, res) => {
    try {
        const {
            index: indexName,
            namespace = "",
            filter = {},
            project = {},
            sort = {},
            limit = 0
        } = req.body || {};

        if (!indexName) {
            return res
                .status(400)
                .json({ status: "error", message: "index is required" });
        }
        const meta = await getIndexMeta(indexName);
        if (!meta) {
            return res
                .status(404)
                .json({ status: "error", message: "index not found" });
        }

        const ns =
            namespace && namespace !== DEFAULT_NAMESPACE ? namespace : "";
        const target = ns
            ? pc.index(indexName).namespace(ns)
            : pc.index(indexName);

        const hasFilter = filter && Object.keys(filter).length > 0;
        const sortKeys =
            sort && typeof sort === "object" ? Object.keys(sort) : [];
        const lim = Math.max(0, parseInt(limit, 10) || 0);

        const queryParams = {
            vector: dummyVector(meta.dimension),
            topK: MAX_TOPK,
            includeMetadata: true,
            includeValues: false
        };
        if (hasFilter) queryParams.filter = filter;

        const result = await target.query(queryParams);
        let matches = result.matches || [];

        if (sortKeys.length) {
            matches = matches.slice().sort((a, b) => {
                for (const k of sortKeys) {
                    const dir = Number(sort[k]) < 0 ? -1 : 1;
                    const av = (a.metadata || {})[k];
                    const bv = (b.metadata || {})[k];
                    if (av === bv) continue;
                    if (av === undefined || av === null) return 1;
                    if (bv === undefined || bv === null) return -1;
                    if (av < bv) return -1 * dir;
                    if (av > bv) return 1 * dir;
                }
                return 0;
            });
        }
        if (lim > 0) matches = matches.slice(0, lim);

        const docs = matches.map((m) => ({
            _id: m.id,
            _score: m.score,
            metadata: applyProject(m.metadata || {}, project)
        }));

        const stamp = new Date().toISOString().replace(/[:.]/g, "-");
        const fname = `${indexName}${ns ? "_" + ns : ""}_${stamp}.json`;
        res.setHeader("Content-Type", "application/json");
        res.setHeader(
            "Content-Disposition",
            `attachment; filename="${fname}"`
        );
        res.send(JSON.stringify(docs, null, 2));
    } catch (err) {
        res.status(500).json({ status: "error", message: err.message });
    }
});

// Delete records. Three modes:
//  - ids: delete specific vector IDs (used by the per-document trash button)
//  - filter: delete every record matching a metadata filter
//  - all: empty filter -> wipe the whole namespace
// Serverless indexes can't delete by metadata filter, so for the filter mode we
// first query the matching IDs (up to MAX_TOPK) and delete those.
// Pass dryRun:true to only count how many would be affected (for the disclaimer).
app.post("/api/delete", async (req, res) => {
    try {
        const {
            index: indexName,
            namespace = "",
            filter = {},
            ids,
            dryRun = false
        } = req.body || {};

        if (!indexName) {
            return res
                .status(400)
                .json({ status: "error", message: "index is required" });
        }

        const meta = await getIndexMeta(indexName);
        if (!meta) {
            return res
                .status(404)
                .json({ status: "error", message: "index not found" });
        }

        const ns =
            namespace && namespace !== DEFAULT_NAMESPACE ? namespace : "";
        const target = ns
            ? pc.index(indexName).namespace(ns)
            : pc.index(indexName);

        // Mode 1: explicit IDs (single-document delete)
        if (Array.isArray(ids) && ids.length > 0) {
            if (!dryRun) {
                for (let i = 0; i < ids.length; i += 1000) {
                    await target.deleteMany(ids.slice(i, i + 1000));
                }
            }
            return res.json({
                status: "success",
                data: { mode: "ids", affected: ids.length, capped: false }
            });
        }

        const hasFilter = filter && Object.keys(filter).length > 0;

        // Mode 3: empty filter -> entire namespace
        if (!hasFilter) {
            let count = 0;
            try {
                const stats = await target.describeIndexStats();
                count = ns
                    ? (stats.namespaces &&
                          stats.namespaces[ns] &&
                          stats.namespaces[ns].recordCount) ||
                      0
                    : stats.totalRecordCount || 0;
            } catch (e) {
                /* best effort */
            }
            if (!dryRun) await target.deleteAll();
            return res.json({
                status: "success",
                data: { mode: "all", affected: count, capped: false }
            });
        }

        // Mode 2: metadata filter -> collect matching IDs, then delete them.
        const result = await target.query({
            vector: dummyVector(meta.dimension),
            topK: MAX_TOPK,
            includeMetadata: false,
            includeValues: false,
            filter
        });
        const matchIds = (result.matches || []).map((m) => m.id);

        if (!dryRun) {
            for (let i = 0; i < matchIds.length; i += 1000) {
                await target.deleteMany(matchIds.slice(i, i + 1000));
            }
        }
        return res.json({
            status: "success",
            data: {
                mode: "filter",
                affected: matchIds.length,
                capped: matchIds.length >= MAX_TOPK
            }
        });
    } catch (err) {
        res.status(500).json({ status: "error", message: err.message });
    }
});

app.listen(PORT, () => {
    // eslint-disable-next-line no-console
    console.log(`Pinecone Compass UI running at http://localhost:${PORT}`);
    // eslint-disable-next-line no-console
    console.log(`Using Pinecone credentials from: ${ENV_PATH}`);
});
