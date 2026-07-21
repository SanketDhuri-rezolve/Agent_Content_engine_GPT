"""Streamlit front end for the Movie Highlight Pipeline's FastAPI/uvicorn
backend. Talks to the same three endpoints api/static/demo.html already
covers (POST /uploads, POST /jobs, GET /jobs/{id}, GET /jobs/{id}/results) —
this is a second, Python-native UI for the same API, not a replacement.

The one JobCreateRequest field with no backend default — source_video_url —
is kept in its own required section; every other field (total_duration_seconds,
segment_count/segment_duration_seconds, sla_target_seconds) has a real
backend-side default/auto-detect path (see models.schemas.JobCreateRequest,
config.Settings), so those are grouped as "automatic" here, each with an
explicit override toggle. Segment sharding specifically offers a choice
between an exact segment_count and a target segment_duration_seconds (the
backend derives segment_count from that via
orchestrator.splitter.compute_segment_count_from_duration) — duration is
usually the more intuitive knob ("shard into ~2-minute chunks") than having
to already know how many segments that implies.

Run with the API already up (`docker-compose up -d --build`, or
`uvicorn api.main:app --port 8001` locally), then:
    streamlit run ui/streamlit_app.py
"""

import time

import requests
import streamlit as st

st.set_page_config(page_title="Movie Highlight Pipeline", page_icon="🎬", layout="centered")

DEFAULT_API_BASE_URL = "http://localhost:8001"
TERMINAL_STATUSES = {"completed", "failed"}
POLL_INTERVAL_SECONDS = 2
STATUS_ICONS = {"completed": "✅", "failed": "❌"}


def api_base_url() -> str:
    return st.session_state.get("api_base_url", DEFAULT_API_BASE_URL).rstrip("/")


def clip_web_url(clip_url: str | None) -> str | None:
    """Mirrors api/static/demo.html's clipWebUrl(): a locally-stored clip_url
    (file:///.../.local_object_storage/highlight_clips/...) is rewritten to
    the API's /clips static mount; a real S3/Azure URL is already directly
    playable as-is."""
    if not clip_url:
        return None
    marker = ".local_object_storage/"
    idx = clip_url.find(marker)
    if idx == -1:
        return None if clip_url.startswith("file://") else clip_url
    return f"{api_base_url()}/clips/{clip_url[idx + len(marker):]}"


def fmt_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def auto_or_manual(label: str, auto_caption: str, key: str, min_value, default_value, step):
    """Renders one JobCreateRequest field as "Auto" (backend default/probe
    applies, nothing sent) unless its override toggle is switched on, in
    which case a number input controls the value sent to POST /jobs."""
    toggle_col, value_col = st.columns([1, 2])
    manual = toggle_col.toggle("Override", key=f"{key}_manual")
    if manual:
        return value_col.number_input(label, min_value=min_value, value=default_value, step=step, key=key)
    value_col.markdown(f"**{label}**  \n:gray[Auto — {auto_caption}]")
    return None


for _key, _default in (("source_video_url", ""), ("detected_duration", None), ("_uploaded_name", None)):
    st.session_state.setdefault(_key, _default)

with st.sidebar:
    st.header("Connection")
    st.text_input(
        "API base URL",
        value=DEFAULT_API_BASE_URL,
        key="api_base_url",
        help="uvicorn/FastAPI backend — docker-compose publishes it on 8001 by default.",
    )
    if st.button("Start a new job"):
        for key in ("job_id", "source_video_url", "detected_duration", "_uploaded_name"):
            st.session_state.pop(key, None)
        st.rerun()

st.title("🎬 Movie Highlight Pipeline")
st.caption("Submit a video, watch it process, see the ranked highlights.")

# ---------------------------------------------------------------------------
# 1. Required — the only JobCreateRequest field with no backend default.
# ---------------------------------------------------------------------------
st.subheader("1. Source video — required")

uploaded_file = st.file_uploader("Upload a video file", type=None)
if uploaded_file is not None and st.session_state._uploaded_name != uploaded_file.name:
    with st.spinner(f"Uploading {uploaded_file.name}…"):
        try:
            resp = requests.post(
                f"{api_base_url()}/uploads",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            st.session_state.source_video_url = data["source_video_url"]
            st.session_state.detected_duration = data["total_duration_seconds"]
            st.session_state._uploaded_name = uploaded_file.name
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")

st.text_input(
    "…or paste a video URL / local path",
    key="source_video_url",
    placeholder="https://example.com/movie.mp4",
)
if st.session_state.detected_duration:
    st.caption(f"Duration auto-detected: {fmt_time(st.session_state.detected_duration)}")

# ---------------------------------------------------------------------------
# 2. Automatic — every other field has a backend default or auto-detect path.
# ---------------------------------------------------------------------------
st.subheader("2. Job parameters — automatic by default")

total_duration_seconds = auto_or_manual(
    "Duration (seconds)",
    (
        f"{fmt_time(st.session_state.detected_duration)} detected from upload"
        if st.session_state.detected_duration
        else "probed via ffprobe from the source"
    ),
    key="total_duration_seconds",
    min_value=1.0,
    default_value=float(st.session_state.detected_duration or 3600.0),
    step=1.0,
)
if total_duration_seconds is None and st.session_state.detected_duration:
    # Already known from the upload — pass it along explicitly so the
    # backend skips a redundant ffprobe call, without showing it as a
    # "manual override" the user asked for.
    total_duration_seconds = st.session_state.detected_duration

st.markdown("**Segment sharding**")
sharding_col, value_col = st.columns([1, 2])
sharding_mode = sharding_col.radio(
    "Sharding mode",
    ["Auto", "By duration", "By count"],
    key="sharding_mode",
    label_visibility="collapsed",
)
segment_count = None
segment_duration_seconds = None
if sharding_mode == "By duration":
    segment_duration_seconds = value_col.number_input(
        "Target seconds per segment", min_value=1.0, value=120.0, step=1.0, key="segment_duration_seconds"
    )
    value_col.caption("segment_count is derived server-side: ceil(total_duration / this value)")
elif sharding_mode == "By count":
    segment_count = value_col.number_input("Segment count", min_value=1, value=4, step=1, key="segment_count")
else:
    value_col.markdown(
        "**Segment count**  \n:gray[Auto — dev placeholder (4), not a confirmed production default, see CLAUDE.md]"
    )

sla_target_seconds = auto_or_manual(
    "SLA target (seconds)",
    "240s backend default",
    key="sla_target_seconds",
    min_value=1,
    default_value=240,
    step=1,
)

st.divider()

submit_disabled = not st.session_state.source_video_url or "job_id" in st.session_state
if st.button("Submit job", type="primary", disabled=submit_disabled):
    payload = {"source_video_url": st.session_state.source_video_url}
    if total_duration_seconds is not None:
        payload["total_duration_seconds"] = total_duration_seconds
    if segment_duration_seconds is not None:
        payload["segment_duration_seconds"] = segment_duration_seconds
    elif segment_count is not None:
        payload["segment_count"] = segment_count
    if sla_target_seconds is not None:
        payload["sla_target_seconds"] = sla_target_seconds

    try:
        resp = requests.post(f"{api_base_url()}/jobs", json=payload, timeout=30)
        resp.raise_for_status()
        st.session_state.job_id = resp.json()["id"]
        st.rerun()
    except requests.RequestException as exc:
        st.error(f"Failed to submit job: {exc}")

if not st.session_state.source_video_url:
    st.caption("Enter a source video above to enable submission.")

# ---------------------------------------------------------------------------
# 3. Status + ranked results
# ---------------------------------------------------------------------------
if "job_id" in st.session_state:
    st.subheader("3. Status")
    try:
        resp = requests.get(f"{api_base_url()}/jobs/{st.session_state.job_id}", timeout=10)
        resp.raise_for_status()
        job = resp.json()
    except requests.RequestException as exc:
        st.error(f"Error polling job: {exc}")
        job = None

    if job:
        status = job["status"]
        st.write(f"{STATUS_ICONS.get(status, '⏳')} **{status}** — {job['total_segments']} segment(s)")

        if status not in TERMINAL_STATUSES:
            time.sleep(POLL_INTERVAL_SECONDS)
            st.rerun()
        elif status == "completed":
            try:
                resp = requests.get(f"{api_base_url()}/jobs/{st.session_state.job_id}/results", timeout=10)
                resp.raise_for_status()
                results = resp.json()["results"]
            except requests.RequestException as exc:
                st.error(f"Error loading results: {exc}")
                results = []

            st.subheader(f"4. Ranked highlights ({len(results)})")
            for r in results:
                rich = r.get("rich_data") or {}
                with st.container(border=True):
                    st.markdown(
                        f"**#{r['rank']}** · {fmt_time(r['start_ts'])}–{fmt_time(r['end_ts'])} "
                        f"· score {r['final_score']:.3f}"
                    )
                    video_url = clip_web_url(r.get("clip_url"))
                    if video_url:
                        st.video(video_url)
                    if rich.get("moment_title"):
                        st.markdown(f"**{rich['moment_title']}**")
                    st.write(rich.get("moment_description") or r.get("transcript_excerpt") or "_(no transcript)_")
                    if r.get("justification"):
                        st.caption(r["justification"])
                    if rich:
                        with st.expander("More details"):
                            st.json(rich)
