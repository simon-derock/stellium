# Test suite for architectural layer validation and dependency boundary enforcement.
# Enforces strict unidirectional architectural boundaries across Stellium packages.

import ast
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _extract_internal_imports(file_path: Path) -> list[str]:
    # Parses a python file and extracts all internal module imports starting with src.
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    internal_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src.") or alias.name == "src":
                    internal_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module.startswith("src.") or node.module == "src"):
                internal_imports.append(node.module)
            elif node.level > 0:
                # Relative import
                internal_imports.append(f"relative_level_{node.level}")
    return internal_imports


def test_domain_models_layer_isolation() -> None:
    # Asserts that domain models in src/models do not import from higher architectural layers.
    models_dir = SRC_DIR / "models"
    for py_file in models_dir.glob("*.py"):
        imports = _extract_internal_imports(py_file)
        for imp in imports:
            # Models must only import from within models or standard packages
            assert not imp.startswith("src.pipelines"), f"{py_file.name} illegally imports {imp}"
            assert not imp.startswith("src.api"), f"{py_file.name} illegally imports {imp}"
            assert not imp.startswith("src.graph"), f"{py_file.name} illegally imports {imp}"
            assert not imp.startswith("src.coprocessor"), f"{py_file.name} illegally imports {imp}"
            assert not imp.startswith("src.llm"), f"{py_file.name} illegally imports {imp}"


def test_coprocessor_layer_isolation() -> None:
    # Asserts that coprocessor compute engine does not depend on pipelines, graph or API.
    coproc_dir = SRC_DIR / "coprocessor"
    for py_file in coproc_dir.glob("*.py"):
        imports = _extract_internal_imports(py_file)
        for imp in imports:
            assert not imp.startswith("src.pipelines"), f"{py_file.name} illegally imports {imp}"
            assert not imp.startswith("src.api"), f"{py_file.name} illegally imports {imp}"
            assert not imp.startswith("src.graph"), f"{py_file.name} illegally imports {imp}"


def test_embeddings_layer_isolation() -> None:
    # Asserts that embeddings module does not depend on pipelines, graph or API.
    py_file = SRC_DIR / "embeddings.py"
    if py_file.exists():
        imports = _extract_internal_imports(py_file)
        for imp in imports:
            assert not imp.startswith("src.pipelines"), f"{py_file.name} illegally imports {imp}"
            assert not imp.startswith("src.api"), f"{py_file.name} illegally imports {imp}"
            assert not imp.startswith("src.graph"), f"{py_file.name} illegally imports {imp}"


def test_graph_layer_isolation() -> None:
    # Asserts that graph database module does not depend on pipelines or API.
    graph_dir = SRC_DIR / "graph"
    for py_file in graph_dir.glob("*.py"):
        imports = _extract_internal_imports(py_file)
        for imp in imports:
            assert not imp.startswith("src.pipelines"), f"{py_file.name} illegally imports {imp}"
            assert not imp.startswith("src.api"), f"{py_file.name} illegally imports {imp}"


def test_no_circular_package_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tests that all core packages can be imported in any sequence without circular dependency errors.
    # monkeypatch restores the original module objects afterwards, so later tests never see two
    # copies of a module (one patched, one still referenced by classes imported earlier).
    modules_to_test = [
        "src.models",
        "src.embeddings",
        "src.coprocessor",
        "src.graph",
        "src.llm",
        "src.pipelines.rag",
        "src.pipelines.graphrag",
        "src.pipelines.agentic",
        "src.evaluate",
        "src.api.main",
    ]
    for mod in modules_to_test:
        parent, _, child = mod.rpartition(".")
        if parent in sys.modules and hasattr(sys.modules[parent], child):
            # Re-importing also rebinds the parent package attribute; restore it afterwards too.
            monkeypatch.setattr(sys.modules[parent], child, getattr(sys.modules[parent], child))
        monkeypatch.delitem(sys.modules, mod, raising=False)
        imported = __import__(mod, fromlist=["*"])
        assert imported is not None


def test_public_interface_contracts() -> None:
    # Verifies that foundational domain models and types are exported cleanly.
    from src.models import AgentState, Chunk, CorpusDoc, PipelineResult, ToolAuditCall

    assert AgentState is not None
    assert Chunk is not None
    assert CorpusDoc is not None
    assert PipelineResult is not None
    assert ToolAuditCall is not None
