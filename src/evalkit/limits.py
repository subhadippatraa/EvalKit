"""Input-size and evidence limits (docs/TARGET-ARCHITECTURE.md §14.2), enforced at the API and
CLI boundaries so an oversized or malformed input fails before any paid judge call."""

from dataclasses import dataclass

# rubric-shape limits (always enforced by the models; not per-instance configurable)
MAX_CRITERIA = 32
MAX_LABELS = 32
MAX_LABEL_CHARS = 64
MAX_DESCRIPTION_CHARS = 2000
MAX_VERSION_CHARS = 128
MAX_SCALE_ABS = 1_000_000
# weights only matter relative to each other; bounding the range keeps the weighted mean
# exactly representable (no overflow to inf, no underflow to 0) -- see Rubric.score
MIN_WEIGHT = 1e-6
MAX_WEIGHT = 1e6

MAX_TAG_CHARS = 128
MAX_REVIEW_COMMENT_CHARS = 16_384
MAX_REVIEWER_CHARS = 128

MAX_INPUT_FILE_BYTES = 8 * 1024 * 1024  # CLI `run --input`
MAX_LIST_LIMIT = 10_000

MAX_EVIDENCE_BYTES = 64 * 1024  # raw judge output kept per failed attempt
MAX_ERROR_CHARS = 2000


@dataclass(frozen=True)
class Limits:
    """Per-Evaluator input limits. Sizes are UTF-8 bytes."""

    max_field_bytes: int = 256 * 1024  # each of prompt / model_output / reference_output / context
    max_metadata_bytes: int = 16 * 1024  # JSON-encoded
    max_tags: int = 32


def truncate_utf8(text: str, max_bytes: int) -> tuple[str, bool]:
    """Cut `text` to at most `max_bytes` UTF-8 bytes without splitting a character."""
    data = text.encode("utf-8", "replace")
    if len(data) <= max_bytes:
        return data.decode("utf-8"), False
    return data[:max_bytes].decode("utf-8", "ignore"), True
