"""Execute this project's plain-Python notebooks without Jupyter dependencies.

Stores real cell outputs in .ipynb files and stops at the first exception.
The combined requirements.txt also provides an interactive Jupyter editor.
Only run notebooks whose code you trust, just as you would a Python script.
"""
import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "standard-library AST executor"
NOTEBOOKS = [
    ROOT / "part_a/Part_A_Data_Foundation.ipynb",
    ROOT / "part_b/Part_B_Customer_Segments_and_Lapse_Prediction.ipynb",
    ROOT / "part_c/Part_C_Support_Ticket_Triage_with_Guardrails.ipynb",
    ROOT / "part_d/Part_D_Policy_RAG_and_Refund_Agent.ipynb",
]


def notebook_source_sha256(notebook):
    """Fingerprint every code/markdown source without mutable execution outputs."""
    sources = [{"cell_type": cell["cell_type"], "id": cell["id"],
                "source": "".join(cell["source"])} for cell in notebook["cells"]]
    payload = json.dumps(sources, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def project_source_sha256():
    """Fingerprint code and dependency pins; generated artifacts/README are excluded."""
    paths = [ROOT / name for name in (
        "generate_data.py", "mock_llm.py", "run_pipeline.py", "sql/queries.sql", "requirements.txt",
        "scripts/check_environment.py", "scripts/run_notebooks.py", "scripts/verify_project.py",
    )]
    for part in "abcd":
        paths.extend((ROOT / f"part_{part}").rglob("*.py"))
    manifest = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(set(paths))}
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_notebook(notebook):
    """Check plain-Python structure and use the official schema when available."""
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] >= 5
    assert isinstance(notebook["metadata"], dict)
    ids = set()
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        assert isinstance(cell["metadata"], dict)
        assert isinstance(cell["source"], (list, str))
        assert isinstance(cell["source"], str) or all(isinstance(line, str) for line in cell["source"])
        assert isinstance(cell["id"], str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell["id"])
        assert cell["id"] not in ids
        ids.add(cell["id"])
        if cell["cell_type"] == "code":
            assert isinstance(cell["outputs"], list)
            ast.parse("".join(cell["source"]))
    try:
        import nbformat
    except ModuleNotFoundError as error:
        if error.name != "nbformat":
            raise
        return "SKIPPED_NOT_INSTALLED"
    nbformat.validate(notebook)
    return "PASSED"


def validate_executed_cells(notebook):
    """Require sequential successful execution and real text output for every cell."""
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code, "Notebook contains no executable code cells."
    assert all(type(cell["execution_count"]) is int for cell in code)
    assert [cell["execution_count"] for cell in code] == list(range(1, len(code) + 1)), \
        "Notebook cells must be executed sequentially from a fresh namespace."
    for cell in code:
        assert cell["outputs"], "Every project code cell must have saved output."
        for output in cell["outputs"]:
            assert output["output_type"] in {"stream", "display_data", "execute_result"}, \
                "Notebook contains an error or unsupported output."
            if output["output_type"] == "stream":
                assert output["name"] in {"stdout", "stderr"}
                assert isinstance(output["text"], (str, list)) and "".join(output["text"]), \
                    "Saved stream output must contain text."
            else:
                assert output["data"] and set(output["data"]) <= {
                    "text/plain", "text/html", "text/markdown", "application/json"}, \
                    "Only text/JSON notebook output is permitted; media must not be committed."
                if output["output_type"] == "execute_result":
                    assert output["execution_count"] == cell["execution_count"]
    return len(code)


def build_execution_evidence(path, notebook, count, schema_validation):
    """Describe a just-executed notebook and the exact source files it used."""
    return {
        "path": path.relative_to(ROOT).as_posix(), "code_cells_executed": count,
        "status": "PASSED", "engine": ENGINE, "python_version": sys.version.split()[0],
        "python_runtime": sys.version, "schema_validation": schema_validation,
        "source_sha256": notebook_source_sha256(notebook),
        "executed_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "project_source_sha256": project_source_sha256(),
    }


def validate_execution_evidence(path, notebook, evidence, current_project_sha256=None):
    """Reject stale sources/outputs or execution claims unsupported by the saved file."""
    schema_validation = validate_notebook(notebook)
    count = validate_executed_cells(notebook)
    assert evidence["path"] == path.relative_to(ROOT).as_posix()
    assert evidence["status"] == "PASSED" and evidence["engine"] == ENGINE
    assert evidence["code_cells_executed"] == count
    assert evidence["schema_validation"] in {"PASSED", "SKIPPED_NOT_INSTALLED"}
    assert isinstance(evidence["python_runtime"], str) and evidence["python_runtime"]
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", evidence["python_version"])
    assert evidence["python_runtime"].split()[0] == evidence["python_version"]
    assert evidence["source_sha256"] == notebook_source_sha256(notebook), \
        f"Notebook sources changed since execution: {path.name}; rerun the notebooks."
    assert evidence["executed_file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), \
        f"Saved notebook content changed since execution: {path.name}; rerun the notebooks."
    source_digest = current_project_sha256 or project_source_sha256()
    assert evidence["project_source_sha256"] == source_digest, \
        "Project code or dependency pins changed since notebook execution; rerun all notebooks."
    return count, schema_validation


def run_notebook(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    # Normalize metadata for standard nbformat 4.5 readers.
    notebook["nbformat_minor"] = 5
    for index, cell in enumerate(notebook["cells"]):
        cell.setdefault("id", hashlib.sha256(f"{path.name}:{index}".encode()).hexdigest()[:12])
    validate_notebook(notebook)
    sources_before_execution = project_source_sha256()
    namespace = {"__name__": "__main__"}
    count = 0
    failed = False
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        if failed:
            cell["execution_count"] = None
            cell["outputs"] = []
            continue
        count += 1
        cell["execution_count"] = count
        cell["outputs"] = []
        stdout, stderr = io.StringIO(), io.StringIO()
        displayed = []

        def display(*values):
            for value in values:
                plain = value.to_string() if hasattr(value, "to_string") else str(value)
                displayed.append({"output_type": "display_data", "metadata": {},
                                  "data": {"text/plain": plain}})

        namespace["display"] = display
        try:
            source = "".join(cell["source"])
            tree = ast.parse(source, filename=f"{path.name}:cell{index + 1}")
            result = None
            with redirect_stdout(stdout), redirect_stderr(stderr):
                # Match notebook behavior: display the final expression's value.
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    expression = tree.body.pop()
                    exec(compile(tree, str(path), "exec"), namespace)
                    result = eval(compile(ast.Expression(expression.value), str(path), "eval"), namespace)
                else:
                    exec(compile(tree, str(path), "exec"), namespace)
            if result is not None:
                namespace["_"] = result
                plain = result.to_string() if hasattr(result, "to_string") else repr(result)
                displayed.append({"output_type": "execute_result", "execution_count": count,
                                  "metadata": {}, "data": {"text/plain": plain}})
        except Exception as error:
            failed = True
            displayed.append({"output_type": "error", "ename": type(error).__name__,
                              "evalue": str(error), "traceback": traceback.format_exc().splitlines()})
            print(f"FAILED {path.relative_to(ROOT)} cell {index + 1}: {error}")
        for name, stream in [("stdout", stdout), ("stderr", stderr)]:
            if stream.getvalue():
                cell["outputs"].append({"output_type": "stream", "name": name,
                                        "text": stream.getvalue().splitlines(keepends=True)})
        cell["outputs"].extend(displayed)
    notebook["metadata"].setdefault("language_info", {})["version"] = sys.version.split()[0]
    schema_validation = validate_notebook(notebook)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError(f"Notebook failed: {path.name}; traceback saved in its cell output.")
    assert validate_executed_cells(notebook) == count
    assert project_source_sha256() == sources_before_execution, \
        "Project code or dependency pins changed during notebook execution; rerun after edits finish."
    print(f"PASSED {path.relative_to(ROOT)}: {count} code cells executed")
    return build_execution_evidence(path, notebook, count, schema_validation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Optional notebook paths; default: all four parts")
    args = parser.parse_args()
    paths = [Path(p).resolve() for p in args.notebooks] if args.notebooks else NOTEBOOKS
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    outcomes = []
    for path in paths:
        os.chdir(ROOT)
        outcomes.append(run_notebook(path))
    output = ROOT / "outputs/notebook_execution_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(outcomes, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
