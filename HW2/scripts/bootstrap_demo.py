"""Build the tiny self-authored TF-IDF demo without downloading textbooks or models."""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ie_agent.config import Settings  # noqa: E402
from ie_agent.knowledge import KnowledgeService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-dir", type=Path, default=ROOT / "knowledge" / "demo")
    args = parser.parse_args()
    target = args.knowledge_dir.resolve()
    index = target / "index" / "tfidf" / "index.pkl"
    if index.exists():
        print(f"Existing index left unchanged: {index}")
        return
    clean = target / "clean"
    if clean.exists():
        parser.error(f"Refusing to overwrite existing content: {clean}")
    shutil.copytree(ROOT / "examples" / "knowledge", clean)
    settings = Settings(_env_file=None, rag_backend="tfidf", knowledge_dir=target)
    result = KnowledgeService(settings).ingest(clean)
    print(f"Built self-authored demo: {result.chunks} chunks; index: {index}")


if __name__ == "__main__":
    main()
