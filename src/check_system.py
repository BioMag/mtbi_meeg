"""Validate dependencies and runtime directories for the pipeline."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os

from packaging.requirements import Requirement
from packaging.version import Version

from config_common import figures_dir, processed_data_dir, raw_data_dir, reports_dir


def read_requirements(path):
    """Read non-empty, non-comment requirement lines."""
    return [
        Requirement(line.strip())
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def check_dependencies(requirements):
    """Raise a clear error for missing or incompatible dependencies."""
    problems = []
    for requirement in requirements:
        try:
            installed = Version(version(requirement.name))
        except PackageNotFoundError:
            problems.append(f"{requirement.name} is not installed")
            continue
        if requirement.specifier and installed not in requirement.specifier:
            problems.append(
                f"{requirement.name} {installed} does not satisfy {requirement.specifier}"
            )
    if problems:
        raise RuntimeError("Dependency check failed:\n- " + "\n- ".join(problems))


def validate_directories(raw, outputs):
    """Require raw input and create output directories."""
    if not os.path.isdir(raw):
        raise ValueError(
            f"The `raw_data_dir` points to a directory that does not exist: {raw}"
        )
    for directory in outputs:
        Path(directory).mkdir(parents=True, exist_ok=True)


def main():
    project_root = Path(__file__).resolve().parents[1]
    check_dependencies(read_requirements(project_root / "requirements.txt"))
    validate_directories(raw_data_dir, (processed_data_dir, figures_dir, reports_dir))
    print("INFO: Success! System requirements are met.")


if __name__ == "__main__":
    main()
