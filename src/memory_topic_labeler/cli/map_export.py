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
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
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
    "assistant": ("Collaboration", "Assistant-led workflows, responses, and execution support"),
    "candidate": ("Candidates", "Potential items being considered for follow-up or promotion"),
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
    "continue": ("Follow-up", "Ongoing follow-up work and continuation of active tasks"),
    "done": ("Completion", "Completed tasks and finalized follow-through"),
    "alert": ("Alerting", "Operational alerting signals and notification events"),
    "health": ("System Health", "Health checks, diagnostics, and reliability monitoring"),
}

FAMILY_ALIASES: dict[str, str] = {
    "dialogue": "Planning & Dialogue",
    "execution state": "Planning & Dialogue",
    "operational state": "Planning & Dialogue",
    "session": "Planning & Dialogue",
    "truths": "Planning & Dialogue",
    "candidates": "Planning & Dialogue",
    "action planning": "Planning & Dialogue",
    "collaboration": "Planning & Dialogue",
    "follow-up": "Planning & Dialogue",
    "completion": "Planning & Dialogue",
    "automation": "Operations",
    "command": "Operations",
    "alerting": "Operations",
    "build": "Operations",
    "publishing": "Operations",
    "sync": "Operations",
    "chartbeat": "Operations",
    "charts": "Operations",
    "vault": "Knowledge",
    "todo": "Knowledge",
    "memory": "Knowledge",
    "storage": "Infrastructure",
    "disk": "Infrastructure",
    "remote ops": "Infrastructure",
    "health": "Infrastructure",
    "system health": "Infrastructure",
    "assistant": "Planning & Dialogue",
}

LABEL_SUFFIX_BLACKLIST = {
    "assistant",
    "automation",
    "candidate",
    "candidates",
    "cluster",
    "command",
    "health",
    "memory",
    "openclaw",
    "session",
    "status",
    "topic",
}

MIN_DOC_LEN = 20


@dataclass
class Cluster:
    id: int
    label: str
    description: str
    size: int
    keywords: list[str]


def normalize_term(term: str) -> str:
    value = term.replace("_", " ").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def family_label(label: str) -> str:
    match = re.match(r"^(.*?)(\d+)$", label.strip())
    if not match:
        return label.strip()
    return match.group(1).strip() or label.strip()


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


def canonical_label(fallback_label: str, fallback_desc: str) -> tuple[str, str]:
    token = normalize_term(fallback_label).lower()
    if token in LABEL_ALIASES:
        return LABEL_ALIASES[token]
    first = token.split(" ", 1)[0]
    if first in LABEL_ALIASES:
        return LABEL_ALIASES[first]
    return fallback_label, fallback_desc


def plain_description(label: str, description: str, keywords: list[str]) -> str:
    mapped = {
        "Automation": "Scheduled and automated jobs, including routine monitoring tasks.",
        "Execution State": "Progress and confidence signals about in-flight work and staging.",
        "Operational State": "System health, run status, and operational evidence from recent activity.",
        "Dialogue": "User and assistant planning conversations, requests, and responses.",
        "Session": "Session metadata and timing context tied to recent interactions.",
        "Disk": "Disk capacity and storage health checks across the system.",
        "Vault": "Knowledge captured from the Obsidian vault, including notes and wiki content.",
        "Todo": "Task planning, follow-up items, and backlog-style reminders.",
        "Build": "Build, test, and validation activity from development workflows.",
        "Truths": "Recurring long-term patterns and distilled durable insights.",
        "Candidates": "Potential items being considered for follow-up or promotion.",
        "Action Planning": "Next actions and decision-oriented planning notes.",
    }
    if label in mapped:
        return mapped[label]

    if "," in description and not re.search(r"\b(is|are|includes|contains|covers|focuses)\b", description.lower()):
        terms = [k.strip() for k in keywords if k.strip()][:3]
        if terms:
            return f"Topics related to {', '.join(terms)}."
        return "Related conversational and operational topics from memory activity."

    cleaned = description.strip().rstrip(".")
    if not cleaned:
        return "Related conversational and operational topics from memory activity."
    return f"{cleaned}."


def choose_label_and_description(words: list[str], topic_id: int) -> tuple[str, str]:
    cleaned = [normalize_term(w) for w in words]
    good = [w for w in cleaned if term_is_good_label(w)]
    label = (good[0] if good else (cleaned[0] if cleaned else f"Topic{topic_id}")).title()
    desc_terms = [w.title() for w in good[1:4]]
    description = ", ".join(desc_terms) if desc_terms else "General themes"
    label, description = canonical_label(label, description)
    return label, plain_description(label, description, cleaned)


def infer_family_name(cluster: Cluster) -> str:
    label_key = normalize_term(cluster.label).lower()
    if label_key in FAMILY_ALIASES:
        return FAMILY_ALIASES[label_key]

    for keyword in cluster.keywords:
        key = normalize_term(keyword).lower()
        if key in FAMILY_ALIASES:
            return FAMILY_ALIASES[key]

    fallback = family_label(cluster.label)
    fallback = re.sub(r"[:\-\s]+$", "", fallback).strip()
    if not fallback:
        return "General Topics"
    fallback_lower = fallback.lower()
    if fallback_lower == label_key or label_key.startswith(f"{fallback_lower}:"):
        return "General Topics"
    return fallback


def uniquify_cluster_labels(clusters: list[Cluster]) -> list[Cluster]:
    label_counts = Counter(c.label for c in clusters)
    label_seen: dict[str, int] = defaultdict(int)
    used_labels: set[str] = set()
    out: list[Cluster] = []

    for cluster in clusters:
        label_seen[cluster.label] += 1
        if label_counts[cluster.label] <= 1:
            used_labels.add(cluster.label)
            out.append(cluster)
            continue

        suffix = ""
        label_tokens = {t for t in re.split(r"[^a-z0-9]+", cluster.label.lower()) if t}
        for keyword in cluster.keywords:
            k = normalize_term(keyword)
            if not k:
                continue
            k_lower = k.lower()
            if k_lower == cluster.label.lower():
                continue
            if not term_is_good_label(k):
                continue
            if k_lower in LABEL_SUFFIX_BLACKLIST:
                continue
            if k_lower in label_tokens:
                continue
            suffix = k.title()
            break
        if not suffix:
            suffix = f"Topic {label_seen[cluster.label]}"

        candidate = f"{cluster.label}: {suffix}"
        if candidate in used_labels:
            for keyword in cluster.keywords:
                k = normalize_term(keyword)
                if not k:
                    continue
                k_lower = k.lower()
                if not term_is_good_label(k):
                    continue
                if k_lower in LABEL_SUFFIX_BLACKLIST:
                    continue
                if k_lower in label_tokens:
                    continue
                candidate_alt = f"{cluster.label}: {k.title()}"
                if candidate_alt not in used_labels:
                    candidate = candidate_alt
                    break
        if candidate in used_labels:
            candidate = f"{cluster.label}: Topic {label_seen[cluster.label]}"

        used_labels.add(candidate)
        out.append(
            Cluster(
                id=cluster.id,
                label=candidate,
                description=cluster.description,
                size=cluster.size,
                keywords=cluster.keywords,
            )
        )

    return out


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
    ctfidf = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
    representation = {
        "Main": KeyBERTInspired(top_n_words=8),
        "Diverse": MaximalMarginalRelevance(diversity=0.35),
    }
    return BERTopic(
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        representation_model=representation,
        umap_model=UMAP(n_neighbors=18, min_dist=0.03, metric="cosine", random_state=42),
        calculate_probabilities=False,
        nr_topics="auto",
        min_topic_size=8,
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


def bertopic_reduce_outliers(topic_model: BERTopic, docs: list[str], topics: list[int], embeddings: np.ndarray) -> list[int]:
    if not any(t == -1 for t in topics):
        return topics
    try:
        reduced = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=embeddings)
        return [int(t) for t in reduced]
    except Exception:
        return topics


def build_hierarchy(clusters: list[Cluster]) -> dict[str, Any]:
    families: dict[str, list[Cluster]] = defaultdict(list)
    for cluster in clusters:
        families[infer_family_name(cluster)].append(cluster)

    family_nodes: list[dict[str, Any]] = []
    for family_name, members in sorted(families.items(), key=lambda kv: sum(c.size for c in kv[1]), reverse=True):
        topic_nodes: list[dict[str, Any]] = []
        for cluster in sorted(members, key=lambda c: c.size, reverse=True):
            kw = [k for k in cluster.keywords if k][:5]
            if not kw:
                kw = ["core"]
            weights = [len(kw) - i for i in range(len(kw))]
            total = sum(weights)
            keyword_nodes = [
                {
                    "id": f"keyword:{cluster.id}:{i}",
                    "label": keyword,
                    "value": max(1, round(cluster.size * (weights[i] / total), 2)),
                }
                for i, keyword in enumerate(kw)
            ]
            topic_nodes.append(
                {
                    "id": f"cluster:{cluster.id}",
                    "clusterId": cluster.id,
                    "label": cluster.label,
                    "description": cluster.description,
                    "children": keyword_nodes,
                }
            )

        family_nodes.append(
            {
                "id": f"family:{family_name.lower().replace(' ', '-')}",
                "label": family_name,
                "children": topic_nodes,
            }
        )

    return {
        "id": "root",
        "label": "Memory",
        "children": family_nodes,
    }


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
            "method": "bertopic+umap+ctfidf+bertopic-outlier-reduction+hierarchy-v3",
            "pointCount": 0,
            "clusterCount": 0,
            "excludedCronChunks": excluded_cron_chunks,
            "excludedShortChunks": excluded_short_chunks,
            "clusters": [],
            "points": [],
            "tree": {"id": "root", "label": "Memory", "value": 0, "children": []},
        }
    else:
        if len(records) > args.max_points:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(len(records), size=args.max_points, replace=False))
            records = [records[i] for i in idx]

        docs = [r["text"] for r in records]
        embeddings = np.stack([r["embedding"] for r in records])

        topic_model, topics = fit_topics(docs, embeddings)
        topics = bertopic_reduce_outliers(topic_model, docs, topics, embeddings)

        topic_counts = Counter(topics)
        finalized_clusters: list[Cluster] = []

        for topic_id, size in sorted(topic_counts.items(), key=lambda kv: kv[1], reverse=True):
            if topic_id == -1:
                continue
            terms = topic_model.get_topic(topic_id) or []
            words = [term for term, _ in terms[:8]]
            label, description = choose_label_and_description(words, topic_id)
            keywords = [normalize_term(w) for w in words]
            finalized_clusters.append(
                Cluster(id=int(topic_id), label=label, description=description, size=int(size), keywords=keywords)
            )

        # Keep Vault/Todo discoverability as dedicated labels when path evidence is strong.
        vault_count = sum(1 for record in records if "obsidian-vault" in record["path"].lower())
        todo_count = sum(1 for record in records if "todo" in record["path"].lower() or "/tasks/" in record["path"].lower())
        if vault_count >= 8 and not any(c.label == "Vault" for c in finalized_clusters):
            finalized_clusters.append(
                Cluster(
                    id=9001,
                    label="Vault",
                    description=plain_description("Vault", "Vault notes and knowledge capture", ["vault", "notes", "wiki"]),
                    size=vault_count,
                    keywords=["vault", "notes", "wiki"],
                )
            )
        if todo_count >= 8 and not any(c.label == "Todo" for c in finalized_clusters):
            finalized_clusters.append(
                Cluster(
                    id=9002,
                    label="Todo",
                    description=plain_description("Todo", "Task tracking and follow-up items", ["todo", "tasks", "follow-up"]),
                    size=todo_count,
                    keywords=["todo", "tasks", "follow-up"],
                )
            )

        finalized_clusters = uniquify_cluster_labels(finalized_clusters)
        finalized_clusters.sort(key=lambda c: c.size, reverse=True)

        cluster_ids = {cluster.id for cluster in finalized_clusters}
        points_clusters = [int(topic) if int(topic) in cluster_ids else 0 for topic in topics]

        umap_model = UMAP(n_neighbors=15, min_dist=0.08, metric="cosine", random_state=42)
        coords = umap_model.fit_transform(embeddings)

        payload = {
            "timestamp": args.timestamp,
            "method": "bertopic+umap+ctfidf+bertopic-outlier-reduction+hierarchy-v3",
            "pointCount": len(records),
            "clusterCount": len(finalized_clusters),
            "excludedCronChunks": excluded_cron_chunks,
            "excludedShortChunks": excluded_short_chunks,
            "clusters": [asdict(c) for c in finalized_clusters],
            "points": [
                {
                    "x": round(float(coords[i][0]), 4),
                    "y": round(float(coords[i][1]), 4),
                    "cluster": int(points_clusters[i]),
                    "path": records[i]["path"],
                    "source": records[i]["source"],
                }
                for i in range(len(records))
            ],
            "tree": build_hierarchy(finalized_clusters),
        }

    if args.output == "-":
        print(json.dumps(payload, indent=2))
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
