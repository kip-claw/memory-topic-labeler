from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from memory_topic_labeler.processor import summarize_topics


def main() -> int:
    parser = argparse.ArgumentParser(description="Memory topic labeler CLI")
    parser.add_argument("--input-json", required=True, help="Path to JSON file containing a list of texts")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        texts = json.load(f)

    summaries = summarize_topics(texts)
    print(json.dumps([asdict(item) for item in summaries], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
