"""How a failed call's evidence is kept (the P0 attempt guarantees), shared by every recorder of
attempts so the single-record path and the run path cannot drift apart.

* Text is made encodable first (a provider can return unpaired surrogates).
* The full text is always hashed; the text itself is kept only for failed calls, truncated to
  MAX_EVIDENCE_BYTES without splitting a character, with a flag saying it was cut.
"""

import hashlib
import json
from typing import Any

from evalkit.limits import MAX_EVIDENCE_BYTES, truncate_utf8


def clip(text: str, max_chars: int) -> str:
    text = text.encode("utf-8", "replace").decode("utf-8")
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def capture_evidence(evidence: Any, *, keep_text: bool) -> tuple[str | None, str | None, bool]:
    """(text, sha256 of the full text, truncated). `text` is None unless `keep_text`."""
    if evidence is None:
        return None, None, False
    try:
        full = json.dumps(evidence, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):  # e.g. circular structure
        full = repr(evidence)
    full = full.encode("utf-8", "replace").decode("utf-8")
    sha = hashlib.sha256(full.encode()).hexdigest()
    if not keep_text:
        return None, sha, False
    text, truncated = truncate_utf8(full, MAX_EVIDENCE_BYTES)
    return text, sha, truncated
