"""Benchmark task registry.

Tasks are deliberately chosen to span the prefill/decode axis, which is what
makes the power-shape ("lumpiness") comparison meaningful:

    mmlu   -> prefill-dominated: long prompt, 1-token answer   -> spiky
    gsm8k  -> decode-dominated:  short prompt, long reasoning  -> flat plateau

The 9 tasks below are the community core (ported from energy-bench's
tasks/, US-MERGE-01). `humaneval_plus` deliberately does NOT live here — it
executes model-generated code in a sandbox, a defensible surface in a
private lab tool but not in a widely installed daemon — so it stays
lab-side, re-registered on top via `register_task` by whatever installs it
(energy-bench's lab layer).
"""

from hmasync_controller.bench.tasks.base import (
    PowerShape,
    Task,
    TaskItem,
    TaskLoadError,
)
from hmasync_controller.bench.tasks.gpqa_diamond import GPQADiamondTask
from hmasync_controller.bench.tasks.gsm8k import GSM8KTask
from hmasync_controller.bench.tasks.gsm8k_platinum import GSM8KPlatinumTask
from hmasync_controller.bench.tasks.hellaswag import HellaSwagTask
from hmasync_controller.bench.tasks.ifeval import IFEvalTask
from hmasync_controller.bench.tasks.longctx_summary import LongctxSummaryTask
from hmasync_controller.bench.tasks.math500 import Math500Task
from hmasync_controller.bench.tasks.mmlu import MMLUTask
from hmasync_controller.bench.tasks.mmlu_redux import MMLUReduxTask

TASK_REGISTRY: dict[str, type[Task]] = {}


class UnknownTaskError(TaskLoadError):
    """Raised when a config names a task that is not registered."""


def register_task(task_cls: type[Task]) -> type[Task]:
    """Register a task class under its `name`.

    Returns the class unchanged, so this doubles as a decorator:

        @register_task
        class MyTask(Task):
            name = "my_task"
            ...

    This is the seam an external package (e.g. energy-bench's lab layer,
    re-registering `humaneval_plus`) uses to extend the registry without
    touching this module.

    Raises:
        ValueError: If `task_cls.name` is already registered.
    """
    if task_cls.name in TASK_REGISTRY:
        raise ValueError(
            f"Task '{task_cls.name}' is already registered to "
            f"{TASK_REGISTRY[task_cls.name].__name__}"
        )
    TASK_REGISTRY[task_cls.name] = task_cls
    return task_cls


def load_task(name: str) -> Task:
    """Instantiate a registered task by name.

    Raises:
        UnknownTaskError: If the name is not in TASK_REGISTRY.
    """
    task_cls = TASK_REGISTRY.get(name)
    if task_cls is None:
        available = ", ".join(sorted(TASK_REGISTRY))
        raise UnknownTaskError(f"Unknown task '{name}'. Available: {available}")
    return task_cls()


for _cls in (
    GSM8KTask,
    GSM8KPlatinumTask,
    MMLUTask,
    MMLUReduxTask,
    GPQADiamondTask,
    Math500Task,
    IFEvalTask,
    HellaSwagTask,
    LongctxSummaryTask,
):
    register_task(_cls)


__all__ = [
    "TASK_REGISTRY",
    "PowerShape",
    "Task",
    "TaskItem",
    "TaskLoadError",
    "UnknownTaskError",
    "register_task",
    "load_task",
]
