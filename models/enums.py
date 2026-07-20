import enum


class JobStatus(str, enum.Enum):
    created = "created"
    queued = "queued"
    sharding = "sharding"
    running_segments = "running_segments"
    reducing = "reducing"
    ranking = "ranking"
    completed = "completed"
    failed = "failed"


class SegmentStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    timeout = "timeout"
