"""Regression tests for the public source layout and dependency direction."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "benchmark"


def _absolute_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imports.append((node.lineno, node.module or ""))
    return imports


def test_single_public_namespace_and_mirrored_tests() -> None:
    assert PACKAGE.is_dir()
    assert PACKAGE.parent.name == "src"
    assert not (ROOT / "benchmark").exists()
    assert not (ROOT / "eeg_mi").exists()
    assert not (ROOT / "deepnet").exists()
    assert (PACKAGE / "__init__.py").is_file()
    for layer in ("research", "shared"):
        assert (PACKAGE / layer / "__init__.py").is_file()
        assert (ROOT / "tests" / layer / "__init__.py").is_file()
    assert (ROOT / "tests" / "core" / "__init__.py").is_file()


def test_dependency_direction_is_acyclic() -> None:
    violations: list[str] = []
    for layer in ("research", "shared"):
        for path in sorted((PACKAGE / layer).rglob("*.py")):
            for line, imported in _absolute_imports(path):
                if not imported.startswith("benchmark"):
                    continue
                allowed = (
                    imported == f"benchmark.{layer}"
                    or imported.startswith(f"benchmark.{layer}.")
                    or (
                        layer == "research"
                        and (
                            imported == "benchmark.shared"
                            or imported.startswith("benchmark.shared.")
                        )
                    )
                )
                if not allowed:
                    violations.append(
                        f"{path.relative_to(ROOT)}:{line} imports {imported}"
                    )
    assert violations == []


def test_source_tree_contains_no_generated_artifacts() -> None:
    banned_directories = {"__pycache__", ".pytest_cache", ".ruff_cache"}
    banned_suffixes = {".pyc", ".pyo", ".pt", ".ckpt"}
    assert not any(
        path.is_dir() and path.name in banned_directories
        for path in PACKAGE.rglob("*")
    )
    assert not any(
        path.is_file() and path.suffix.lower() in banned_suffixes
        for path in PACKAGE.rglob("*")
    )
