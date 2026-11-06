"""Check the Python, SQLite and combined pinned dependencies before reproduction."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_TOOLS = {"ipykernel", "notebook", "nbformat", "nbclient"}


def inspect_environment():
    requirements = ROOT / "requirements.txt"
    packages = []
    seen = set()
    for line in requirements.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if not match:
            raise ValueError(f"Expected one exact package pin: {line}")
        name, expected = match.groups()
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in seen:
            raise ValueError(f"Duplicate dependency: {name}")
        seen.add(normalized)
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        packages.append({"name": name, "expected": expected, "installed": installed,
                         "group": "notebooks" if name in NOTEBOOK_TOOLS else "analytics",
                         "status": "PASSED" if installed == expected else "MISSING" if installed is None else "VERSION_MISMATCH"})
    python_ok = sys.version_info >= (3, 12)
    sqlite_ok = sqlite3.sqlite_version_info >= (3, 25, 0)
    analytics_ok = all(p["status"] == "PASSED" for p in packages if p["group"] == "analytics")
    notebooks_ok = all(p["status"] == "PASSED" for p in packages if p["group"] == "notebooks")
    return {"status": "PASSED" if python_ok and sqlite_ok and analytics_ok and notebooks_ok else "INCOMPLETE",
            "python": sys.version.split()[0], "python_requirement": ">=3.12", "python_passed": python_ok,
            "sqlite": sqlite3.sqlite_version, "sqlite_requirement": ">=3.25", "sqlite_passed": sqlite_ok,
            "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
            "analytics_passed": analytics_ok, "notebook_tools_passed": notebooks_ok,
            "packages": packages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytics-only", action="store_true",
                        help="Check pipeline prerequisites while reporting any absent interactive notebook tools.")
    args = parser.parse_args()
    result = inspect_environment()
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs/environment_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Python {result['python']}; SQLite {result['sqlite']}")
    print("Analytics dependencies:", "PASSED" if result["analytics_passed"] else "INCOMPLETE")
    print("Interactive notebook dependencies:", "PASSED" if result["notebook_tools_passed"] else "INCOMPLETE")
    for package in result["packages"]:
        if package["status"] != "PASSED":
            print(f"  {package['name']}=={package['expected']}: {package['status']}")
    passed = result["python_passed"] and result["sqlite_passed"] and result["analytics_passed"]
    if not args.analytics_only:
        passed = passed and result["notebook_tools_passed"]
    if not passed:
        print("Install the combined requirements.txt in this Python environment, then rerun this check.")
        return 1
    print("ENVIRONMENT CHECK PASSED" + (" (analytics only)" if args.analytics_only else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
