"""Repo-wide invariants that nothing else checks.

Notebooks and Streamlit pages are never imported by the rest of the suite, so
a syntax error in them ships silently. These are cheap and catch that.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

PYTHON_FILES = sorted(
    p for p in REPO.rglob("*.py")
    if ".git" not in p.parts
    and ".github" not in p.parts
    and "build" not in p.parts
)


def test_there_are_python_files_to_check():
    assert len(PYTHON_FILES) > 10


@pytest.mark.parametrize(
    "path", PYTHON_FILES, ids=lambda p: str(p.relative_to(REPO)),
)
def test_file_parses(path):
    """Every Python file in the repo must be syntactically valid."""
    ast.parse(path.read_text(), filename=str(path))


JOB_FILES = sorted((REPO / "jobs").glob("*.py"))


@pytest.mark.parametrize("path", JOB_FILES, ids=lambda p: p.name)
def test_job_entry_points_have_no_import_side_effects(path):
    """Importing a job module must not install packages or hit the network.

    The entry points used to shell out to pip at module scope, before
    argparse and once per process under any multi-process launch.
    """
    tree = ast.parse(path.read_text(), filename=str(path))

    for node in tree.body:
        # Skip definitions: what happens inside main() runs on invocation,
        # not on import, and calling the installer from there is the fix.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        dumped = ast.dump(node)
        assert "check_call" not in dumped, (
            f"{path.name} runs a subprocess at import time"
        )
        assert "ensure_runtime_requirements" not in dumped, (
            f"{path.name} installs requirements at import time"
        )


@pytest.mark.parametrize("path", JOB_FILES, ids=lambda p: p.name)
def test_job_entry_points_do_not_add_src_to_the_path(path):
    """Adding both the root and src/ makes one module importable under two names.

    ``import config`` and ``import src.config`` would then resolve to two
    different module objects, each with its own registry state.
    """
    source = path.read_text()
    assert '"src"' not in source and "'src'" not in source, (
        f"{path.name} appears to add src/ to sys.path"
    )
