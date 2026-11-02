"""Execute this project's plain-Python notebooks without Jupyter dependencies.

Stores real cell outputs in .ipynb files and stops at the first exception.
For an interactive editor, optionally install requirements-notebooks.txt.
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
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "part_a/Part_A_Data_Foundation.ipynb",
    ROOT / "part_b/Part_B_Customer_Segments_and_Lapse_Prediction.ipynb",
    ROOT / "part_c/Part_C_Support_Ticket_Triage_with_Guardrails.ipynb",
    ROOT / "part_d/Part_D_Policy_RAG_and_Refund_Agent.ipynb",
]


def validate_notebook(notebook):
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] >= 5
    assert isinstance(notebook["metadata"], dict)
    ids = set()
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        assert isinstance(cell["metadata"], dict)
        assert isinstance(cell["source"], (list, str))
        assert cell["id"] not in ids
        ids.add(cell["id"])
        if cell["cell_type"] == "code":
            assert isinstance(cell["outputs"], list)
            ast.parse("".join(cell["source"]))


def run_notebook(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    # Normalize metadata for standard nbformat 4.5 readers.
    notebook["nbformat_minor"] = 5
    for index, cell in enumerate(notebook["cells"]):
        cell.setdefault("id", hashlib.sha256(f"{path.name}:{index}".encode()).hexdigest()[:12])
    validate_notebook(notebook)
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
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError(f"Notebook failed: {path.name}; traceback saved in its cell output.")
    print(f"PASSED {path.relative_to(ROOT)}: {count} code cells executed")
    return {"path": str(path.relative_to(ROOT)), "code_cells_executed": count, "status": "PASSED"}


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
