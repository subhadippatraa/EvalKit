"""The system under test for the quickstart: a stand-in for your model, RAG pipeline or agent.

`candidate` is the "new version": it gets the answer right except when the sum is at least 40,
where it is off by one (a regression the comparison should catch). A real target would call your
application; it receives ONLY the prompt and context, never the reference answer.
"""

import re

from evalkit.targets import TargetInput


def candidate(inp: TargetInput) -> str:
    a, b = map(int, re.findall(r"\d+", inp.prompt))
    total = a + b
    return str(total + 1 if total >= 60 else total)
