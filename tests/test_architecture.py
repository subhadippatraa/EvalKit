"""Import-direction rules (design 2.1, adapted to the flat module layout): the deterministic core
imports nothing above it, SDKs are confined to the two provider modules, and only the interface
layer may depend on everything."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "evalkit"

# module -> the evalkit modules it may import (top-level imports; lazy in-function ones are checked
# separately because they are how optional SDKs and cycles are kept out)
ALLOWED = {
    "errors": {"redact"},
    "redact": set(),
    "limits": set(),
    "safejson": set(),
    "hashing": set(),
    "stats": set(),
    "evidence": {"limits"},
    "failures": {"evidence", "limits", "redact"},
    "models": {"errors", "limits"},
    "llm": {"failures"},
    "judge": {"models"},
    "judge_eval": {"errors", "failures", "judge", "llm", "models"},
    "datasets": {"errors", "hashing", "limits", "models", "safejson"},
    "runs": {"datasets", "errors", "evidence", "failures", "hashing", "limits", "models", "redact"},
    "calls": {"failures", "llm", "runs"},
    "targets": {"calls", "datasets", "errors", "failures", "limits", "llm", "runs"},
    "evaluators": {
        "calls",
        "errors",
        "failures",
        "judge",
        "judge_eval",
        "llm",
        "models",
        "runs",
        "safejson",
    },
    "bedrock": {"errors", "failures", "judge_eval", "llm"},
    "bedrock_openai": {"failures", "judge_eval", "llm", "safejson"},
    "env": {"errors", "llm"},
    "runlock": {"errors"},
    "analysis": {"runs", "stats"},
    "compare": {"analysis", "errors", "stats"},
    "calibration": {"errors", "limits", "stats"},
    "judgecheck": {
        "calls",
        "errors",
        "evaluators",
        "failures",
        "llm",
        "models",
        "runs",
        "safejson",
    },
    "report": {"analysis", "calibration", "compare", "errors"},
}
STORES = {"dataset_store", "run_store", "analysis_store", "store", "migrations"}
ENGINE = {"engine"}
INTERFACE = {"cli", "cli_platform", "kit", "__init__", "evaluator"}
SDKS = {"boto3", "botocore", "openai"}


def imports(path: Path):
    """(top-level evalkit imports, all imported top-level package names, lazy evalkit imports)."""
    tree = ast.parse(path.read_text())
    top, lazy, external = set(), set(), set()

    def names(node):
        if isinstance(node, ast.Import):
            return [a.name for a in node.names]
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            return [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        return []

    for node in ast.walk(tree):
        for name in names(node):
            root = name.split(".")[0]
            if root == "evalkit":
                parts = name.split(".")
                if len(parts) > 1:
                    (top if _module_level(tree, node) else lazy).add(parts[1])
            else:
                external.add(root)
    return top, external, lazy


def _module_level(tree, target):
    return (
        any(target is n for n in tree.body)
        or any(target is c for n in tree.body if isinstance(n, ast.If) for c in n.body)
        and not _in_typechecking(tree, target)
    )


def _in_typechecking(tree, target):
    for n in tree.body:
        if isinstance(n, ast.If) and "TYPE_CHECKING" in ast.dump(n.test):
            if any(target is c for c in ast.walk(n)):
                return True
    return False


@pytest.mark.parametrize("module", sorted(ALLOWED))
def test_the_core_modules_import_only_what_the_layering_allows(module):
    top, _, _ = imports(SRC / f"{module}.py")
    assert top <= ALLOWED[module] | {module}, f"{module} imports {sorted(top - ALLOWED[module])}"


def test_every_module_is_classified():
    modules = {p.stem for p in SRC.glob("*.py")}
    known = set(ALLOWED) | STORES | ENGINE | INTERFACE
    assert modules <= known, f"unclassified modules: {sorted(modules - known)}"


def test_provider_sdks_are_imported_only_by_the_two_provider_modules():
    for path in SRC.glob("*.py"):
        _, external, lazy = imports(path)
        offenders = external & SDKS
        if path.stem in ("bedrock", "bedrock_openai"):
            continue
        assert not offenders, f"{path.name} imports {sorted(offenders)}"


def test_the_pure_analysis_layer_never_touches_the_engine_stores_or_cli():
    forbidden = STORES | ENGINE | {"cli", "cli_platform", "kit", "evaluator"}
    for module in (
        "analysis",
        "compare",
        "calibration",
        "report",
        "stats",
        "judgecheck",
        "evaluators",
        "targets",
        "calls",
    ):
        top, _, _ = imports(SRC / f"{module}.py")
        assert not top & forbidden, f"{module} imports {sorted(top & forbidden)}"


def test_only_the_interface_layer_imports_the_engine():
    for path in SRC.glob("*.py"):
        top, _, lazy = imports(path)
        if path.stem in INTERFACE or path.stem == "engine":
            continue
        assert "engine" not in top | lazy, path.name


def test_optional_dependencies_are_never_imported_at_module_level():
    for path in SRC.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            )
            assert not any(n.split(".")[0] in ("jsonschema", "referencing") for n in names), (
                path.name
            )


def test_the_core_import_does_not_load_provider_sdks():
    import subprocess
    import sys

    code = (
        "import sys, evalkit, evalkit.cli, evalkit.engine, evalkit.report;"
        "bad = {'boto3', 'botocore', 'openai', 'jsonschema'} & set(sys.modules);"
        "assert not bad, bad"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
