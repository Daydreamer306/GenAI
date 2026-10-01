"""Run repeatable verification and save a redacted terminal transcript."""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ie_agent.config import Settings  # noqa: E402
from ie_agent.run_artifacts import RunArtifactWriter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Also call configured MiniMax")
    parser.add_argument(
        "--live-rag",
        action="store_true",
        help="Also send retrieved textbook excerpts; requires explicit user consent",
    )
    args = parser.parse_args()
    if args.live_rag and not args.live:
        parser.error("--live-rag requires --live")
    os.chdir(ROOT)
    settings = Settings()
    if args.live and (settings.model_provider != "minimax" or not settings.has_model_key):
        parser.error("Live verification requires a configured MiniMax key")
    destination = settings.runs_dir / (
        "verification-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    )
    destination.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1")
    redact = RunArtifactWriter(settings).redact
    results = []
    commands = [(["-m", "unittest", "discover", "-s", "tests", "-v"], 0)]
    for scenario in (
        "repair",
        "pass",
        "reject",
        "solver-error",
        "reviewer-error",
        "invalid-review",
        "tool-error",
    ):
        commands.append(
            (
                ["-m", "ie_agent.cli", "demo", "--scenario", scenario],
                0 if scenario in {"repair", "pass"} else 2,
            )
        )
    commands.append((["-m", "ie_agent.cli", "reproduce"], 0))
    if args.live:
        if args.live_rag:
            question = "根据教材计算概率 [0.5,0.5] 的信息熵，并引用来源。"
        else:
            question = json.dumps(
                {"tool_name": "entropy", "arguments": {"probabilities": [0.5, 0.5]}}
            )
        commands.extend(
            [
                (
                    [
                        "-m",
                        "ie_agent.cli",
                        "ask",
                        question,
                        "--planner",
                        "rules",
                    ],
                    0,
                ),
                (["-m", "ie_agent.cli", "benchmark"], 0),
            ]
        )
    with (destination / "terminal.log").open("w", encoding="utf-8") as log:
        for arguments, expected in commands:
            label = "python " + " ".join(arguments)
            print(f"Running: {label}", flush=True)
            log.write(f"\n$ {label}\n")
            log.flush()
            process = subprocess.Popen(
                [sys.executable, *arguments],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                safe = redact(line)
                log.write(safe)
                log.flush()
                print(safe, end="", flush=True)
            code = process.wait()
            results.append(
                {
                    "command": label,
                    "exit_code": code,
                    "expected_exit_code": expected,
                    "passed": code == expected,
                }
            )
    report = {
        "live_tests_enabled": args.live,
        "live_textbook_test_enabled": args.live_rag,
        "passed": all(r["passed"] for r in results),
        "checks": results,
    }
    (destination / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verification artifacts: {destination}", flush=True)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
