from memory_topic_labeler.cli.map_export import (
    Cluster,
    build_hierarchy,
    is_excluded_memory_path,
    term_is_good_label,
)


def test_dreaming_paths_are_excluded_from_public_memory_maps() -> None:
    assert is_excluded_memory_path("memory/dreaming/light/2026-07-19.md")
    assert is_excluded_memory_path("memory/.dreams/session-corpus/2026-07-19.txt")
    assert not is_excluded_memory_path("memory/2026-07-19.md")
    assert not is_excluded_memory_path("MEMORY.md")


def test_pronouns_and_demonstratives_are_not_topic_labels() -> None:
    for term in ("you", "your", "yours", "our", "their", "this", "those"):
        assert not term_is_good_label(term)

    assert term_is_good_label("Diagnostics")


def test_hierarchy_root_has_a_value_for_stats_consumers() -> None:
    tree = build_hierarchy(
        [Cluster(id=1, label="Diagnostics", description="Health checks.", size=7, keywords=["pi"])]
    )

    assert tree["value"] == 7
