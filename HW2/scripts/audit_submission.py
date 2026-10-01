"""Check staged content for secrets, private data, symlinks, and oversized files."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secrets-file", type=Path)
    args = parser.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root).split(b"\0")
    known_secrets = []
    if args.secrets_file:
        for line in args.secrets_file.read_text().splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip().endswith("API_KEY"):
                value = value.strip().strip("\"'")
                if value:
                    known_secrets.append(value.encode())
    token = re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}")
    issues = []
    total = 0
    count = 0
    for entry in entries:
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, object_id, _ = metadata.split()
        path = raw_path.decode()
        count += 1
        forbidden = (
            any(
                part in {".venv", "node_modules", "__pycache__", ".ruff_cache"}
                for part in Path(path).parts
            )
            or path.startswith(("HW2/knowledge/", "HW2/runs/"))
            or (Path(path).name.startswith(".env") and Path(path).name != ".env.example")
        )
        if forbidden:
            issues.append({"file": path, "issue": "private/generated path staged"})
        if mode == b"120000":
            issues.append({"file": path, "issue": "symlink staged"})
        data = subprocess.check_output(["git", "cat-file", "blob", object_id.decode()], cwd=root)
        total += len(data)
        if len(data) > 10 * 1024 * 1024:
            issues.append({"file": path, "issue": "file exceeds submission 10 MiB policy"})
        if token.search(data) or any(secret in data for secret in known_secrets):
            issues.append({"file": path, "issue": "possible credential"})
        if path.startswith("HW2/artifacts/") and path.endswith("/evidence.json"):
            if json.loads(data).get("sources"):
                issues.append({"file": path, "issue": "original textbook excerpts staged"})
    report = {"passed": not issues and count > 0, "files": count, "bytes": total, "issues": issues}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
