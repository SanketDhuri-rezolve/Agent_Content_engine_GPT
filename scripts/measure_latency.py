#!/usr/bin/env python3
"""Step 3 latency measurement harness.

Submits a real job through the running API, polls it to completion, and
reports per-stage and cumulative timing — this produces the real numbers
behind the 240s budget estimate (see CLAUDE.md's "Segment count is NOT
decided" note: the 8-vs-12 decision should be made from THIS harness's output
against real GPU-backed adapters, not a guess).

Assumes the stack is already running (`docker-compose up -d --build`, or with
workers/segment_worker/Dockerfile.gpu swapped in once Step 2's real adapters
are GPU-validated). Per-stage timing is reconstructed from `docker compose
logs`, which requires:
  1. This script to be run from a machine with the `docker compose` CLI and
     access to this project's compose stack (i.e. typically the same host
     that ran `docker-compose up`).
  2. config.logging.configure_logging() to have been wired into every
     process (already true — see orchestrator/celery_app.py, api/main.py) so
     every stage.started/stage.completed/stage.failed log line is a single
     parseable JSON object rather than structlog's default human-readable
     ConsoleRenderer text.

If log collection isn't possible (docker compose unavailable, or a non-
compose deployment like RunPod), pass --skip-log-collection to still get the
harness-measured cumulative wall-clock and the job's own created_at/
completed_at from the API, just without the per-stage breakdown.

Usage:
    python scripts/measure_latency.py \\
        --source-video-url https://example.com/movie.mp4 \\
        --total-duration-seconds 7200 \\
        --segment-count 8 \\
        --sla-target-seconds 240

Compare 8 vs 12 (or any other count) by simply re-running with a different
--segment-count and diffing the reports — this script does not itself decide
that value (see config.Settings.provisional_dev_segment_count's docstring:
that decision is explicitly deferred to real measurements like this one).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every service whose logs carry per-stage structlog lines for a job. Deliberately
# excludes postgres/redis/qdrant (infra, no stage.* lines) — api is included since
# it logs the job_submission stage.completed line.
LOGGED_SERVICES = [
    "api",
    "orchestrator",
    "segment_worker",
    "span_builder",
    "scorer",
    "reducer",
    "ranker",
]

TERMINAL_STATUSES = {"completed", "failed"}


@dataclass
class StageLogLine:
    stage: str
    event: str
    elapsed_seconds: float | None
    segment_id: str | None
    sequence_index: int | None
    job_id: str | None
    extra: dict = field(default_factory=dict)


def submit_job(
    api_base_url: str,
    source_video_url: str,
    total_duration_seconds: float | None,
    segment_count: int | None,
    sla_target_seconds: int | None,
) -> dict:
    payload = {"source_video_url": source_video_url}
    if total_duration_seconds is not None:
        payload["total_duration_seconds"] = total_duration_seconds
    if segment_count is not None:
        payload["segment_count"] = segment_count
    if sla_target_seconds is not None:
        payload["sla_target_seconds"] = sla_target_seconds

    response = httpx.post(f"{api_base_url}/jobs", json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


def poll_job(api_base_url: str, job_id: str, poll_interval_seconds: float, timeout_seconds: float) -> dict:
    """Polls GET /jobs/{id} until status is terminal or timeout_seconds elapses.
    Returns the last JobResponse seen — callers must check `status` themselves;
    a timeout is NOT an error here (a job stuck past the SLA budget is exactly
    the finding this harness exists to surface, not something to hide behind
    an exception)."""
    deadline = time.monotonic() + timeout_seconds
    last: dict = {}
    while time.monotonic() < deadline:
        response = httpx.get(f"{api_base_url}/jobs/{job_id}", timeout=10.0)
        response.raise_for_status()
        last = response.json()
        if last.get("status") in TERMINAL_STATUSES:
            return last
        time.sleep(poll_interval_seconds)
    return last


def fetch_job_results(api_base_url: str, job_id: str) -> dict:
    response = httpx.get(f"{api_base_url}/jobs/{job_id}/results", timeout=30.0)
    response.raise_for_status()
    return response.json()


def collect_stage_logs(job_id: str, since_seconds_ago: int) -> list[StageLogLine]:
    """Runs `docker compose logs` for every LOGGED_SERVICES entry and parses
    every JSON-shaped line whose job_id matches. Returns [] (not an error) if
    docker compose isn't available or the compose project isn't running —
    callers should treat an empty result as "per-stage breakdown unavailable",
    not as "the job produced no stage transitions"."""
    try:
        result = subprocess.run(
            [
                "docker", "compose", "logs",
                "--no-color", "--no-log-prefix",
                f"--since={since_seconds_ago}s",
                *LOGGED_SERVICES,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"warning: could not collect docker compose logs ({exc}); per-stage breakdown skipped", file=sys.stderr)
        return []

    if result.returncode != 0:
        print(
            f"warning: `docker compose logs` exited {result.returncode}: "
            f"{result.stderr.strip()[:500]}; per-stage breakdown skipped",
            file=sys.stderr,
        )
        return []

    # Two passes are needed: span_builder/scorer's task payloads genuinely
    # don't carry job_id (only segment_id/sequence_index — see
    # models.schemas.SegmentWorkerOutput and the candidate-span dict shape),
    # so those stages' log lines never have job_id either, by design. Only
    # segment_worker's lines carry BOTH job_id and segment_id — pass 1 uses
    # those to learn which segment_ids belong to this job_id; pass 2 then
    # matches every other line by segment_id membership in that set.
    records: list[dict] = []
    for raw_line in result.stdout.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        # Celery hijacks the root logger with its own "[timestamp: LEVEL/
        # ProcessName] " prefix ahead of every message regardless of
        # config.logging's own formatter — so a structlog JSON line looks
        # like "[2026-...: WARNING/ForkPoolWorker-8] {"job_id": ...}", not a
        # bare JSON object. Locate the embedded object rather than assuming
        # the whole line is JSON (this also naturally skips Celery's own
        # banners/task-received lines, which have no "{" at all).
        brace_index = raw_line.find("{")
        if brace_index == -1:
            continue
        try:
            record = json.loads(raw_line[brace_index:])
        except json.JSONDecodeError:
            continue
        if record.get("event", "").startswith("stage."):
            records.append(record)

    job_segment_ids = {
        record["segment_id"]
        for record in records
        if record.get("job_id") == job_id and record.get("segment_id")
    }

    lines: list[StageLogLine] = []
    for record in records:
        belongs_to_job = record.get("job_id") == job_id or (
            record.get("job_id") is None and record.get("segment_id") in job_segment_ids
        )
        if not belongs_to_job:
            continue
        lines.append(
            StageLogLine(
                stage=record.get("stage", "unknown"),
                event=record["event"],
                elapsed_seconds=record.get("elapsed_seconds"),
                segment_id=record.get("segment_id"),
                sequence_index=record.get("sequence_index"),
                job_id=job_id,
                extra={
                    k: v
                    for k, v in record.items()
                    if k not in {"stage", "event", "elapsed_seconds", "segment_id", "sequence_index", "job_id", "level", "timestamp"}
                },
            )
        )
    return lines


def aggregate_stage_timing(lines: list[StageLogLine]) -> dict[str, dict]:
    """Groups stage.completed lines by `stage` and computes count/min/mean/max
    elapsed_seconds. A stage that fans out across N segments (segment_worker,
    span_builder, scorer) naturally has N entries here — the spread between
    min and max is itself informative (segment-to-segment variance)."""
    by_stage: dict[str, list[float]] = {}
    for line in lines:
        if line.event != "stage.completed" or line.elapsed_seconds is None:
            continue
        by_stage.setdefault(line.stage, []).append(line.elapsed_seconds)

    return {
        stage: {
            "count": len(values),
            "min_seconds": min(values),
            "mean_seconds": statistics.fmean(values),
            "max_seconds": max(values),
        }
        for stage, values in sorted(by_stage.items())
    }


def print_report(
    job: dict,
    harness_wall_clock_seconds: float,
    stage_timing: dict[str, dict],
    results: dict | None,
    sla_target_seconds: float,
) -> None:
    print()
    print("=" * 72)
    print(f"Job {job.get('id')} — status: {job.get('status')}")
    print("=" * 72)
    print(f"Segments: {job.get('total_segments')}")
    print(f"SLA target: {sla_target_seconds:.0f}s")
    print()

    print(f"Harness-measured wall-clock (submit -> terminal status): {harness_wall_clock_seconds:.2f}s")
    created_at, completed_at = job.get("created_at"), job.get("completed_at")
    if created_at and completed_at:
        print(f"DB-recorded wall-clock (created_at -> completed_at):     {created_at} -> {completed_at}")
    print()

    status = job.get("status")
    if status == "completed":
        verdict = "PASS" if harness_wall_clock_seconds <= sla_target_seconds else "FAIL"
        print(f"SLA verdict: {verdict} ({harness_wall_clock_seconds:.2f}s vs {sla_target_seconds:.0f}s budget)")
    elif status == "failed":
        print(f"SLA verdict: FAIL (job status=failed after {harness_wall_clock_seconds:.2f}s, before reaching a ranked result)")
    else:
        # Never call this a PASS: an elapsed time under budget means nothing
        # if the job hadn't actually finished when we stopped polling — that
        # is a timeout/hang finding, not a fast success.
        print(
            f"SLA verdict: INCOMPLETE (status={status!r} after {harness_wall_clock_seconds:.2f}s — "
            f"did not reach a terminal status within --timeout-seconds; re-run with a longer timeout)"
        )
    print()

    if stage_timing:
        print("Per-stage timing (from docker compose logs):")
        print(f"  {'stage':<20} {'count':>6} {'min(s)':>10} {'mean(s)':>10} {'max(s)':>10}")
        for stage, stats in stage_timing.items():
            print(
                f"  {stage:<20} {stats['count']:>6} {stats['min_seconds']:>10.4f} "
                f"{stats['mean_seconds']:>10.4f} {stats['max_seconds']:>10.4f}"
            )
    else:
        print("Per-stage timing: unavailable (see warnings above — pass --skip-log-collection to silence)")
    print()

    if results is not None:
        print(f"Ranked highlights returned: {len(results.get('results', []))}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-video-url", required=True)
    parser.add_argument("--total-duration-seconds", type=float, default=None)
    parser.add_argument("--segment-count", type=int, default=None)
    parser.add_argument("--sla-target-seconds", type=float, default=None)
    parser.add_argument("--api-base-url", default="http://localhost:8001")
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--skip-log-collection",
        action="store_true",
        help="Skip the docker-compose-logs-based per-stage breakdown (e.g. non-compose deployments).",
    )
    args = parser.parse_args()

    print(f"Submitting job: {args.source_video_url}")
    start = time.monotonic()
    job = submit_job(
        args.api_base_url,
        args.source_video_url,
        args.total_duration_seconds,
        args.segment_count,
        args.sla_target_seconds,
    )
    job_id = job["id"]
    sla_target_seconds = job.get("sla_target_seconds") or args.sla_target_seconds or 240
    print(f"Job {job_id} submitted (total_segments={job.get('total_segments')}), polling...")

    final_job = poll_job(args.api_base_url, job_id, args.poll_interval_seconds, args.timeout_seconds)
    harness_wall_clock_seconds = time.monotonic() - start

    if final_job.get("status") not in TERMINAL_STATUSES:
        print(
            f"warning: job did not reach a terminal status within --timeout-seconds={args.timeout_seconds} "
            f"(last observed status: {final_job.get('status')!r}) — this IS a latency finding, not a script bug",
            file=sys.stderr,
        )

    stage_timing: dict[str, dict] = {}
    if not args.skip_log_collection:
        # +5s pad: the job may have started fractionally before `start` was
        # recorded (network round-trip to submit_job) — better to over-fetch
        # logs than silently miss the earliest stage.started lines.
        since_seconds_ago = int(harness_wall_clock_seconds) + 5
        stage_logs = collect_stage_logs(job_id, since_seconds_ago)
        stage_timing = aggregate_stage_timing(stage_logs)

    results = None
    if final_job.get("status") == "completed":
        results = fetch_job_results(args.api_base_url, job_id)

    print_report(final_job, harness_wall_clock_seconds, stage_timing, results, sla_target_seconds)


if __name__ == "__main__":
    main()
