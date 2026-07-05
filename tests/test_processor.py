from memory_topic_labeler.processor import summarize_topics


def test_empty_input_returns_empty() -> None:
    assert summarize_topics([]) == []
