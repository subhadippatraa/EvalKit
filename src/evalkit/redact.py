"""Best-effort scrubbing of secrets/identifiers from text that is stored or printed.

Applied to every error message evalkit persists or prints (provider error text can embed IAM
principal ARNs, account IDs, keys or presigned-URL signatures). Defensive, not exhaustive:
it is not applied to judged content or to judge reasoning, which is evidence.
"""

import re

_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"arn:aws[a-z-]*:[a-z0-9-]*:[a-z0-9-]*:\d{12}:[^\s\"',;)]+"), "arn:aws:<redacted>"),
    (re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[A-Z0-9]{16}\b"), "<aws-key-id>"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "<api-key>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=\-]{8,}"), "Bearer <redacted>"),
    (re.compile(r"(?i)(account(?:\s*id)?[\s:=#]+)\d{12}\b"), r"\1<redacted>"),
    (
        re.compile(
            r"(?i)\b(x-amz-signature|x-amz-security-token|x-amz-credential|aws_secret_access_key"
            r"|aws_session_token|api[_-]?key|x-api-key|authorization|secret|password|token)"
            r"(\"?\s*[:=]\s*\"?)([^\s\"'&,;]{4,})"
        ),
        r"\1\2<redacted>",
    ),
]


def scrub(text: str) -> str:
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text
