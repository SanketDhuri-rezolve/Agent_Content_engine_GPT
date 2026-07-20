"""Fast manual sanity check for the Postgres wiring — NOT part of the pytest
suite (see tests/conftest.py for that; this script is for a human to run by
hand and eyeball the output).

Usage:
    docker-compose up -d postgres redis
    cp .env.example .env   # if you haven't already
    python scripts/smoke_test.py

What it does:
    1. Ensures the schema exists (Base.metadata.create_all) against whatever
       DATABASE_URL is configured (config.settings.Settings.database_url).
    2. Round-trips one fake Job + Segment through orchestrator.state's
       create_job / create_segments and prints what came back.

Signature note: orchestrator/state.py and orchestrator/splitter.py landed with
`create_job(session, source_video_url, sla_target_seconds, total_segments) -> Job`,
`create_segments(session, job_id, segment_plan: list[dict]) -> list[Segment]`,
and `compute_segment_plan(total_duration_seconds, segment_count, overlap_seconds) -> list[dict]`
respectively — segment_plan must be built via compute_segment_plan first, it is
not something create_segments derives on its own from raw duration/count.
"""

from models.db import Base, get_engine, session_scope
from orchestrator.splitter import compute_segment_plan
from orchestrator.state import create_job, create_segments


def main() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"Schema ensured on {engine.url!s}")

    with session_scope() as session:
        segment_plan = compute_segment_plan(
            total_duration_seconds=120.0,
            segment_count=1,
            overlap_seconds=5.0,
        )

        job = create_job(
            session,
            source_video_url="https://example.com/fake-movie.mp4",
            sla_target_seconds=240,
            total_segments=len(segment_plan),
        )
        print(f"Created job: id={job.id} status={job.status} total_segments={job.total_segments}")

        segments = create_segments(session, job.id, segment_plan)
        for segment in segments:
            print(
                f"Created segment: id={segment.id} sequence_index={segment.sequence_index} "
                f"start_ts={segment.start_ts} end_ts={segment.end_ts} "
                f"overlap_start={segment.overlap_start} overlap_end={segment.overlap_end} "
                f"status={segment.status}"
            )

    print("Smoke test OK.")


if __name__ == "__main__":
    main()
