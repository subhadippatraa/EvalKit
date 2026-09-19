import pytest

from evalkit.store import SQLiteStore

CRITERIA = {"correctness": "Is it factually correct?", "clarity": "Is it clear?"}


def judged(**scores: int) -> dict:
    """Raw judge payload for the given criterion scores."""
    return {name: {"reasoning": f"{name} looks fine", "score": s} for name, s in scores.items()}


class FakeJudge:
    """Test Judge: returns scripted raw payloads or raises scripted exceptions, in order."""

    provider = "fake"
    model = "fake-model"
    temperature = 0.0
    prompt_version = "fakeversion1"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def judge(self, prompt, model_output, reference_output, rubric):
        self.calls.append((prompt, model_output, reference_output, rubric))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(tmp_path / "evalkit.db")
    yield s
    s.close()
