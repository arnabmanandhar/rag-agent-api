"""Ask a question through retrieval, grounded generation, and fallback."""

import argparse
import json

from app.inference.orchestrator import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the local TaskFlow knowledge base")
    parser.add_argument("question", help="Question to answer")
    args = parser.parse_args()
    try:
        response = answer_question(args.question)
    except Exception as exc:
        parser.exit(status=1, message=f"Query failed: {type(exc).__name__}: {exc}\n")
    print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
