from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from bertopic import BERTopic


@dataclass
class ClusterSummary:
    cluster_id: int
    label: str
    description: str
    size: int


def summarize_topics(texts: Iterable[str]) -> list[ClusterSummary]:
    docs = [text.strip() for text in texts if text and text.strip()]
    if not docs:
        return []

    topic_model = BERTopic(calculate_probabilities=False, verbose=False)
    topics, _ = topic_model.fit_transform(docs)
    topic_info = topic_model.get_topic_info()

    summaries: list[ClusterSummary] = []
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue
        terms = topic_model.get_topic(topic_id) or []
        words = [term for term, _ in terms[:4]]
        label = words[0].title() if words else f"Topic{topic_id}"
        description = ", ".join(word.title() for word in words[1:4]) if len(words) > 1 else "General"
        size = int(row["Count"])
        summaries.append(
            ClusterSummary(cluster_id=topic_id, label=label, description=description, size=size)
        )

    summaries.sort(key=lambda item: item.size, reverse=True)
    return summaries
