"""Every third-party module the package imports is a declared runtime dependency.

The dev and eval groups install packages the shipped wheel does not carry, and
macOS arm64 pulls more in transitively (parakeet-mlx → librosa → scipy). Either
hides an undeclared import from CI; 0.7.0 shipped to Windows without scipy that
way and diarization died on the first clustering call (found 2026-11-03). This
reads the imports from the source instead of trusting the environment.
"""

import ast
import re
import sys
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "stenograf"

# Import names that differ from the distribution that provides them, or that a
# declared dependency provides on purpose rather than by accident.
_PROVIDED_BY = {
    "PySide6": {"pyside6-essentials"},
    "onnxruntime": {"onnxruntime", "onnxruntime-directml"},
    # onnx-asr[hub] pins the HuggingFace download path (see pyproject.toml).
    "huggingface_hub": {"onnx-asr"},
    # The MLX runtime under parakeet-mlx / mlx-lm (macOS arm64 only).
    "mlx": {"parakeet-mlx", "mlx-lm"},
}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared() -> set[str]:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return {
        _normalize(re.match(r"[A-Za-z0-9._-]+", dep).group()) for dep in project["dependencies"]
    }


def _imported() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(_SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                found.setdefault(name.split(".")[0], path)
    return found


def test_every_third_party_import_is_a_runtime_dependency():
    declared = _declared()
    missing = []
    for module, path in _imported().items():
        if module in sys.stdlib_module_names or module in {"__future__", "stenograf"}:
            continue
        providers = _PROVIDED_BY.get(module, {_normalize(module)})
        if not providers & declared:
            missing.append(f"{module} (first seen in {path.relative_to(_ROOT)})")
    assert not missing, "imported but not in [project] dependencies: " + ", ".join(missing)
