from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from umap import UMAP

CRON_SENDER_PATTERNS = [
    re.compile(r'"sender"\s*:\s*"cron(?:\b|[^"]*)"', re.IGNORECASE),
    re.compile(r'"sender_id"\s*:\s*"cron(?:\b|[^"]*)"', re.IGNORECASE),
    re.compile(r'"label"\s*:\s*"cron(?:\b|[^"]*)"', re.IGNORECASE),
    re.compile(r'Sender \(untrusted metadata\).*?"label"\s*:\s*"cron', re.IGNORECASE | re.DOTALL),
]

NOISE_PATTERNS = [
    re.compile(r"```json.*?```", re.IGNORECASE | re.DOTALL),
    re.compile(r"Conversation info \(untrusted metadata\):", re.IGNORECASE),
    re.compile(r"Sender \(untrusted metadata\):", re.IGNORECASE),
    re.compile(r"Session Key:\s*[^\n]+", re.IGNORECASE),
    re.compile(r"Session ID:\s*[^\n]+", re.IGNORECASE),
    re.compile(r"timestamp\s*:\s*[^\n]+", re.IGNORECASE),
    re.compile(r"\b(path|source)\s*:\s*[^\n]+", re.IGNORECASE),
]

DOMAIN_STOPWORDS = {
    "assistant",
    "candidate",
    "chat",
    "chunk",
    "chunks",
    "command",
    "commands",
    "confidence",
    "conversation",
    "cron",
    "dreaming",
    "evidence",
    "exited",
    "health",
    "json",
    "memory",
    "message",
    "metadata",
    "ran",
    "run",
    "sender",
    "session",
    "staged",
    "status",
    "telegram",
    "timestamp",
    "untrusted",
}

BAD_LABEL_TERMS = {
    "the",
    "and",
    "to",
    "if",
    "is",
    "are",
    "was",
    "were",
    "for",
    "from",
    "with",
    "user",
    "can",
}

LABEL_ALIASES: dict[str, tuple[str, str]] = {
    "you": ("Dialogue", "User-assistant dialogue and planning"),
    "staged": ("Execution State", "Workflow state, confidence, and staging"),
    "status": ("Operational State", "System status, checks, and evidence"),
    "sync": ("Sync", "Repository and file synchronization"),
    "publish": ("Publishing", "Release, deploy, and publish workflow"),
    "release": ("Publishing", "Release, deploy, and publish workflow"),
    "build": ("Build", "Build, checks, and validation"),
    "lint": ("Build", "Build, checks, and validation"),
    "test": ("Build", "Build, checks, and validation"),
    "chart": ("Charts", "Stats and visualization updates"),
    "stats": ("Charts", "Stats and visualization updates"),
    "cluster": ("Topic Modeling", "Topic extraction and semantic labeling"),
    "topic": ("Topic Modeling", "Topic extraction and semantic labeling"),
    "bertopic": ("Topic Modeling", "Topic extraction and semantic labeling"),
    "label": ("Topic Modeling", "Topic extraction and semantic labeling"),
    "embedding": ("Embeddings", "Embedding generation and semantic space"),
    "embeddings": ("Embeddings", "Embedding generation and semantic space"),
    "cron": ("Automation", "Scheduled tasks and cron operations"),
    "scheduler": ("Automation", "Scheduled tasks and cron operations"),
    "openclaw": ("OpenClaw", "OpenClaw operations and tooling"),
    "agent": ("Agents", "Agent behavior and execution flow"),
    "memory": ("Memory", "Memory indexing and retrieval pipeline"),
    "todo": ("Todo", "Task tracking and follow-up items"),
    "tasks": ("Todo", "Task tracking and follow-up items"),
    "vault": ("Vault", "Vault notes and knowledge capture"),
    "sqlite": ("Storage", "SQLite and data persistence operations"),
    "database": ("Storage", "SQLite and data persistence operations"),
    "ssh": ("Remote Ops", "Remote execution and host orchestration"),
    "tailnet": ("Remote Ops", "Remote execution and host orchestration"),
    "svelte": ("Frontend", "Svelte UI and page composition"),
    "ui": ("Frontend", "Svelte UI and page composition"),
}

OUTLIER_REASSIGN_MIN_SIM = 0.32
MIN_DOC_LEN = 20


def normalize_term(term: str) -> str:
    value = term.replace("_", " ").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def term_is_good_label(term: str) -> bool:
    lower = term.lower()
    if not lower or lower in BAD_LABEL_TERMS:
        return False
    if re.fullmatch(r"[0-9]+", lower):
        return False
    if len(lower) < 3:
        return False
    if not re.search(r"[a-z]", lower):
        return False
    return True


def canonical_label(words: list[str], fallback_label: str, fallback_desc: str) -> tuple[str, str]:
    terms = [normalize_term(w).lower() for w in words]
    probe = [normalize_term(fallback_label).lower(), *terms]
    for token in probe:
        if token in LABEL_ALIASES:
            return LABEL_ALIASES[token]
        first = token.split(" ", 1)[0]
        if first in LABEL_ALIASES:
            return LABEL_ALIASES[first]
    return fallback_label, fallback_desc


def choose_label_and_description(words: list[str], topic_id: int) -> tuple[str, str]:
    cleaned = [normalize_term(w) for w in words]
    good = [w for w in cleaned if term_is_good_label(w)]
    label = (good[0] if good else (cleaned[0] if cleaned else f"Topic{topic_id}")).title()
    desc_terms = [w.title() for w in good[1:4]]
    description = ", ".join(desc_terms) if desc_terms else "General themes"
    return canonical_label(cleaned, label, description)


@dataclass
class Cluster:
    id: int
    label: str
    description: str
    size: int
    keywords: list[str]


def is_cron_sender_chunk(text: str) -> bool:
    return any(p.search(text) for p in CRON_SENDER_PATTERNS)


def clean_text(text: str) -> str:
    value = text
    for pattern in NOISE_PATTERNS:
        value = pattern.sub(" ", value)
    value = re.sub(r"\b[A-Za-z0-9_./-]+\.(json|md|py|ts|svelte|sh)\b", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def build_topic_model() -> BERTopic:
    vectorizer = CountVectorizer(
        stop_words=sorted(set(ENGLISH_STOP_WORDS).union(DOMAIN_STOPWORDS)),
        ngram_range=(1, 2),
        min_df=2,
        token_pattern=r"(?u)\\b[a-zA-Z][a-zA-Z\\-]{1,}\\b",
    )
    return BERTopic(
        vectorizer_model=vectorizer,
        umap_model=UMAP(n_neighbors=18, min_dist=0.03, metric="cosine", random_state=42),
        calculate_probabilities=False,
        nr_topics=10,
        min_topic_size=14,
        top_n_words=8,
        verbose=False,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export semantic map JSON from OpenClaw sqlite")
    p.add_argument("--sqlite", required=True, help="Path to openclaw-agent sqlite")
    p.add_argument("--timestamp", required=True)
    p.add_argument("--output", default="-", help="Output JSON path or - for stdout")
    p.add_argument("--max-points", type=int, default=700)
    return p.parse_args()


def fit_topics(docs: list[str], embeddings: np.ndarray) -> tuple[BERTopic, list[int]]:
    topic_model = build_topic_model()
    try:
        topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
        return topic_model, [int(t) for t in topics]
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower():
            raise
        fallback = BERTopic(calculate_probabilities=False, verbose=False)
        topics, _ = fallback.fit_transform(docs, embeddings=embeddings)
        return fallback, [int(t) for t in topics]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return values / denom


def reduce_outliers(topics: list[int], embeddings: np.ndarray) -> list[int]:
    topic_arr = np.array(topics, dtype=np.int32)
    outlier_idx = np.where(topic_arr == -1)[0]
    if outlier_idx.size == 0:
        return topics

    non_outlier_labels = sorted({int(t) for t in topic_arr.tolist() if int(t) != -1})
    if len(non_outlier_labels) < 2:
        return topics

    centroids: list[np.ndarray] = []
    labels: list[int] = []
    for label in non_outlier_labels:
        mask = topic_arr == label
        if int(mask.sum()) < 4:
            continue
        centroids.append(embeddings[mask].mean(axis=0))
        labels.append(label)

    if not centroids:
        return topics

    centroid_mat = normalize_rows(np.stack(centroids))
    emb_norm = normalize_rows(embeddings)

    for idx in outlier_idx.tolist():
        sims = centroid_mat @ emb_norm[idx]
        best = int(np.argmax(sims))
        if float(sims[best]) >= OUTLIER_REASSIGN_MIN_SIM:
            topic_arr[idx] = labels[best]

    return [int(t) for t in topic_arr.tolist()]


def merge_cluster_families(clusters: list[Cluster]) -> tuple[list[Cluster], dict[int, int]]:
    grouped: dict[str, list[Cluster]] = defaultdict(list)
    remap: dict[int, int] = {}

    for cluster in clusters:
        grouped[cluster.label].append(cluster)

    merged: list[Cluster] = []
    for label, members in grouped.items():
        if label == "Outlier":
            outlier = members[0]
            merged.append(outlier)
            remap[outlier.id] = outlier.id
            continue

        merged_id = min(member.id for member in members)
        total_size = sum(member.size for member in members)
        desc = Counter(member.description for member in members).most_common(1)[0][0]

        kw_counter: Counter[str] = Counter()
        for member in members:
            for kw in member.keywords:
                key = normalize_term(kw).lower()
                if key:
                    kw_counter[key] += 1

        keywords = [key for key, _ in kw_counter.most_common(8)]
        merged.append(Cluster(id=merged_id, label=label, description=desc, size=total_size, keywords=keywords))

        for member in members:
            remap[member.id] = merged_id

    merged.sort(key=lambda c: c.size, reverse=True)
    return merged, remap


def main() -> int:
    args = parse_args()

    conn = sqlite3.connect(args.sqlite)
    rows = conn.execute(
        """
        select id, path, source, text, embedding
        from memory_index_chunks
        where embedding is not null and source = 'memory'
        """
    ).fetchall()
    conn.close()

    records: list[dict[str, Any]] = []
    excluded_cron_chunks = 0
    excluded_short_chunks = 0
    for chunk_id, path, source, text, embedding_json in rows:
        raw_text = str(text or "")
        if is_cron_sender_chunk(raw_text):
            excluded_cron_chunks += 1
            continue
        cleaned = clean_text(raw_text)
        if len(cleaned) < MIN_DOC_LEN:
            excluded_short_chunks += 1
            continue
        try:
            embedding = np.array(json.loads(embedding_json), dtype=np.float32)
        except Exception:
            continue
        if embedding.ndim != 1 or embedding.size < 2:
            continue
        records.append(
            {
                "chunkId": str(chunk_id),
                "path": str(path or ""),
                "source": str(source or ""),
                "text": cleaned,
                "embedding": embedding,
            }
        )

    if not records:
        payload = {
            "timestamp": args.timestamp,
            "method": "bertopic+umap+ctfidf+stable-labels+outlier-reassign+family-merge",
            "pointCount": 0,
            "clusterCount": 0,
            "excludedCronChunks": excluded_cron_chunks,
            "excludedShortChunks": excluded_short_chunks,
            "clusters": [],
            "points": [],
        }
    else:
        if len(records) > args.max_points:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(len(records), size=args.max_points, replace=False))
            records = [records[i] for i in idx]

        docs = [r["text"] for r in records]
        embeddings = np.stack([r["embedding"] for r in records])

        topic_model, topics = fit_topics(docs, embeddings)
        topics = reduce_outliers(topics, embeddings)

        umap_model = UMAP(n_neighbors=15, min_dist=0.08, metric="cosine", random_state=42)
        coords = umap_model.fit_transform(embeddings)

        topic_counts = Counter(topics)
        raw_clusters: list[Cluster] = []

        for topic_id, size in sorted(topic_counts.items(), key=lambda kv: kv[1], reverse=True):
            if topic_id == -1:
                label = "Outlier"
                description = "Unclustered fragments"
                keywords = ["outlier"]
            else:
                terms = topic_model.get_topic(topic_id) or []
                words = [term for term, _ in terms[:8]]
                label, description = choose_label_and_description(words, topic_id)
                keywords = [normalize_term(w) for w in words]
            raw_clusters.append(
                Cluster(
                    id=int(topic_id),
                    label=label,
                    description=description,
                    size=int(size),
                    keywords=keywords,
                )
            )

        clusters, topic_remap = merge_cluster_families(raw_clusters)

        payload = {
            "timestamp": args.timestamp,
            "method": "bertopic+umap+ctfidf+stable-labels+outlier-reassign+family-merge",
            "pointCount": len(records),
            "clusterCount": len(clusters),
            "excludedCronChunks": excluded_cron_chunks,
            "excludedShortChunks": excluded_short_chunks,
            "clusters": [asdict(c) for c in clusters],
            "points": [
                {
                    "x": round(float(coords[i][0]), 4),
                    "y": round(float(coords[i][1]), 4),
                    "cluster": int(topic_remap.get(int(topics[i]), int(topics[i]))),
                    "path": records[i]["path"],
                    "source": records[i]["source"],
                }
                for i in range(len(records))
            ],
        }

    if args.output == "-":
        print(json.dumps(payload, indent=2))
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
