"""What a bare `pip install async-energy-controller` gets you (US-MERGE-06).

Two acceptance criteria live here, and both are about the SHAPE of the
install rather than any single behaviour:

1. **Dependency audit** — the declared base dependencies are exactly
   httpx, pydantic, pydantic-settings, pynvml (shipped as the `nvidia-ml-py`
   distribution), pyarrow, huggingface-hub, langdetect, and their transitive
   closure stays free of the heavyweights the merge program deliberately kept
   out. Operator decision 2 of `tasks/prd-controller-merge.md`: everything
   ships in the BASE install, not a `[bench]` extra — which only works if the
   base install stays small enough to be defensible.

2. **Base-install import check, on a no-GPU box** — the whole package tree
   imports cleanly with `pynvml` unimportable, `get_profiler()` degrades
   instead of raising, and `bench quick`'s NVML requirement surfaces as a
   stated `NvmlUnavailableError`. Run in a SUBPROCESS: `pynvml` is importable
   in this venv (and `tests/test_bench_quick.py` installs a MagicMock for it
   in `sys.modules`), so the no-GPU environment can only be simulated in a
   fresh interpreter that never imported it.

The import check also asserts what must NOT be in `sys.modules` afterwards.
That is the real regression guard: a stray top-level `import numpy` inside a
ported module would keep passing every functional test on a dev box that has
numpy installed, and only break on the stranger's machine this program exists
to serve.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# The agreed set, from tasks/prd-controller-merge.md's success metrics:
# "Controller base deps: exactly httpx, pydantic, pydantic-settings, pynvml,
# pyarrow, huggingface-hub, langdetect."
#
# `pynvml` is the IMPORT name; the distribution that provides it is
# `nvidia-ml-py` (NVIDIA's own official binding — the separate `pynvml`
# distribution on PyPI is a deprecated third-party fork). The audit is over
# distribution names, so that is the name listed here; the no-GPU check below
# is what pins the import name.
EXPECTED_BASE_DEPS = {
    "httpx",
    "pydantic",
    "pydantic-settings",
    "nvidia-ml-py",
    "pyarrow",
    "huggingface-hub",
    "langdetect",
}

# Packages whose presence anywhere in the base install's transitive closure
# would mean the merge went wrong. Each has a reason:
#   numpy/nltk/absl-py/rouge-score — the `rouge-score` chain US-MERGE-06
#     removed (bench/tasks/rouge_l.py replaced it).
#   duckdb — the run index is the lab's analysis store (GROUND TRUTH).
#   fastapi/uvicorn/starlette — the dashboard and collector are lab-side.
#   docker — engine adapters here are ATTACH MODE ONLY; no launch path.
#   pandas/datasets/dill/multiprocess — what `datasets` would drag in;
#     bench/tasks/base.py reads parquet via pyarrow instead.
#   energy-bench — the whole point of the program: no dependency back.
#
# NOT listed, deliberately: `six`. It arrives via `langdetect`, which IS one
# of the seven agreed dependencies, so it is part of the agreed set's own
# cost — not a leftover of the rouge-score chain it also happened to belong
# to. Removing rouge-score dropped absl-py, nltk and numpy; six stayed, and
# that is correct.
FORBIDDEN_IN_CLOSURE = {
    "numpy", "nltk", "absl-py", "rouge-score",
    "duckdb", "fastapi", "uvicorn", "starlette", "docker",
    "pandas", "datasets", "dill", "multiprocess",
    "energy-bench",
}


def _normalize(name: str) -> str:
    """PEP 503 normalized distribution name."""
    import re

    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_dependencies() -> list[str]:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["dependencies"]


def _requirement_name(spec: str) -> str:
    from packaging.requirements import Requirement

    return _normalize(Requirement(spec).name)


class TestDependencyAudit:
    def test_base_dependencies_are_exactly_the_agreed_set(self):
        declared = {_requirement_name(s) for s in _declared_dependencies()}
        assert declared == {_normalize(n) for n in EXPECTED_BASE_DEPS}

    def test_every_base_dependency_is_pinned(self):
        """"The canonical pin list" in pyproject.toml's own words: a floating
        dependency makes an energy measurement irreproducible across boxes."""
        from packaging.requirements import Requirement

        for spec in _declared_dependencies():
            specifiers = list(Requirement(spec).specifier)
            assert len(specifiers) == 1 and specifiers[0].operator == "==", spec

    def test_pynvml_is_importable_from_the_declared_distribution(self):
        """Ties the distribution name in the audit to the import name the
        code actually uses, so neither can be renamed silently."""
        assert _normalize("nvidia-ml-py") in {
            _requirement_name(s) for s in _declared_dependencies()
        }
        dist = metadata.distribution("nvidia-ml-py")
        top_level = {
            Path(f).parts[0].removesuffix(".py")
            for f in (dist.files or [])
            if not str(f).startswith(("..", "nvidia_ml_py-"))
        }
        assert "pynvml" in top_level

    def test_bench_ships_in_the_base_install_not_an_extra(self):
        """Operator decision 2: single download, no `[bench]` extra."""
        with PYPROJECT.open("rb") as fh:
            project = tomllib.load(fh)["project"]
        extras = project.get("optional-dependencies", {})
        assert "bench" not in extras
        # `gpu` survives only as a back-compat no-op alias (US-MERGE-02).
        assert extras.get("gpu") == []

    def test_transitive_closure_excludes_the_heavyweights(self):
        """The audit that actually caught something: `rouge-score` was a
        one-line dependency that pulled four more packages behind it."""
        from packaging.requirements import Requirement

        seen: set[str] = set()
        queue = [_requirement_name(s) for s in _declared_dependencies()]
        while queue:
            name = queue.pop()
            if name in seen:
                continue
            seen.add(name)
            try:
                requires = metadata.requires(name) or []
            except metadata.PackageNotFoundError:
                pytest.skip(f"{name} is not installed in this environment")
            for spec in requires:
                req = Requirement(spec)
                # Skip extras-gated and environment-gated requirements: neither
                # is part of a plain `pip install` on this platform.
                if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                    continue
                queue.append(_normalize(req.name))

        assert not (seen & {_normalize(n) for n in FORBIDDEN_IN_CLOSURE}), sorted(
            seen & {_normalize(n) for n in FORBIDDEN_IN_CLOSURE}
        )

    def test_no_module_imports_a_forbidden_package(self):
        """Source-level twin of the closure check: an import of something
        that merely happens to be installed on a dev box (pytest's own deps,
        a leftover from before a removal) would slip past `sys.modules`
        assertions but not past the AST."""
        import ast

        forbidden_imports = {
            "numpy", "pandas", "duckdb", "fastapi", "uvicorn", "starlette",
            "docker", "datasets", "rouge_score", "nltk", "energy_bench",
        }
        offenders: list[str] = []
        for path in sorted((REPO_ROOT / "hmasync_controller").rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module] if node.module else []
                else:
                    continue
                for name in names:
                    if name and name.split(".")[0] in forbidden_imports:
                        offenders.append(f"{path.relative_to(REPO_ROOT)}: {name}")
        assert not offenders, offenders


# --- the no-GPU base install ------------------------------------------------

# Every distribution a real `pip install async-energy-controller` does NOT
# put on the box. Blocking them in the probe is what makes this a base-install
# check rather than a "works on my dev box" check: numpy in particular IS
# installed in this venv (a leftover of the removed rouge-score chain) and
# pyarrow imports it opportunistically when present, so a plain
# `"numpy" not in sys.modules` assertion would fail here while passing
# nowhere it matters.
_ABSENT_ON_A_BASE_INSTALL = [
    "numpy", "pandas", "duckdb", "fastapi", "uvicorn", "starlette",
    "docker", "datasets", "rouge_score", "nltk", "absl", "energy_bench",
]

_NO_GPU_PROBE = r"""
import json, importlib, pkgutil, sys, tempfile

BLOCKED = set(json.loads(sys.argv[1])) | {"pynvml"}


# A box with no NVIDIA binding and nothing but the seven base dependencies
# installed: every blocked name raises ImportError, exactly as it would where
# the distribution was never installed.
class _BlockedFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError(f"simulated base install: {fullname} is not installed")
        return None


assert not (BLOCKED & set(sys.modules)), sorted(BLOCKED & set(sys.modules))
sys.meta_path.insert(0, _BlockedFinder())

import hmasync_controller
import hmasync_controller.cli  # noqa: F401

imported = []
for mod in pkgutil.walk_packages(hmasync_controller.__path__, "hmasync_controller."):
    importlib.import_module(mod.name)
    imported.append(mod.name)

for name in sorted(BLOCKED):
    try:
        importlib.import_module(name)
    except ImportError:
        continue
    raise AssertionError(f"{name} imported despite the block")

# 1. The profiler degrades instead of raising.
from hmasync_controller.profiler import NVMLProfiler, get_profiler

profiler = get_profiler()
assert not isinstance(profiler, NVMLProfiler), type(profiler).__name__

# 2. The bench sampler constructs (no I/O at construction) and states the
#    problem when actually asked to measure.
from hmasync_controller.bench.sampler import LocalNvmlSampler, NvmlUnavailableError

try:
    LocalNvmlSampler()._ensure_handle()
except NvmlUnavailableError as e:
    assert "pynvml is not importable" in str(e), str(e)
else:
    raise AssertionError("expected NvmlUnavailableError on a no-GPU box")

# 3. Scoring needs no GPU and no numpy: the parquet reader every task loads
#    through, and the scorer that replaced rouge-score, both work here.
import pyarrow as pa
import pyarrow.parquet as pq

with tempfile.NamedTemporaryFile(suffix=".parquet") as fh:
    pq.write_table(pa.table({"question": ["q"], "answer": ["a #### 7"]}), fh.name)
    rows = pq.read_table(fh.name).to_pylist()
assert rows == [{"question": "q", "answer": "a #### 7"}], rows

from hmasync_controller.bench.tasks import TASK_REGISTRY, load_task
from hmasync_controller.bench.tasks.base import TaskItem

item = TaskItem(item_id="longctx_summary:0", prompt="p", target="the lazy dog")
assert load_task("longctx_summary").score("the lazy dog", item) == 1.0
assert load_task("gsm8k_platinum").score("so the answer is #### 7", TaskItem(
    item_id="gsm8k_platinum:0", prompt="p", target="7")) is True

print(json.dumps({
    "modules": sorted(imported),
    "profiler": type(profiler).__name__,
    "tasks": sorted(TASK_REGISTRY),
}))
"""


@pytest.fixture(scope="module")
def no_gpu_probe() -> dict:
    """Import the whole package in a fresh interpreter that has neither an
    NVIDIA binding nor anything outside the seven base dependencies."""
    import json

    proc = subprocess.run(
        [sys.executable, "-c", _NO_GPU_PROBE, json.dumps(_ABSENT_ON_A_BASE_INSTALL)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180, check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestBaseInstallOnANoGpuBox:
    def test_every_module_imports(self, no_gpu_probe):
        modules = no_gpu_probe["modules"]
        # Spot-check the ones that would plausibly reach for NVML or a
        # lab-only dependency at import time.
        for expected in (
            "hmasync_controller.cli",
            "hmasync_controller.profiler",
            "hmasync_controller.powercap",
            "hmasync_controller.nvml_reader",
            "hmasync_controller.bench.quick",
            "hmasync_controller.bench.sampler",
            "hmasync_controller.bench.tasks.longctx_summary",
            "hmasync_controller.bench.metrics.compute",
        ):
            assert expected in modules

    def test_profiler_degrades_instead_of_raising(self, no_gpu_probe):
        assert no_gpu_probe["profiler"] in {"SmiProfiler", "NullProfiler"}

    def test_all_nine_community_tasks_registered_without_a_gpu(self, no_gpu_probe):
        assert no_gpu_probe["tasks"] == sorted([
            "gpqa_diamond", "gsm8k", "gsm8k_platinum", "hellaswag", "ifeval",
            "longctx_summary", "math500", "mmlu", "mmlu_redux",
        ])
        # Operator decision 3: humaneval_plus executes model-generated code
        # and stays lab-side, never in a widely installed daemon.
        assert "humaneval_plus" not in no_gpu_probe["tasks"]
