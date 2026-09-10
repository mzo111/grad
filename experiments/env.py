"""Machine and library facts, collected during a run rather than written by hand.

The benchmark's numbers only mean something next to the machine that produced them, and a
hand-written hardware line goes stale silently. This is the same idea as recording the
command beside a figure: collect it, print it with the table, and let a stale value be
visible instead of plausible.

Everything here degrades to a null rather than raising: a benchmark must not fail because
`/proc` looks different on someone else's box.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _mem_total_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return None


def _torch_facts() -> dict[str, Any]:
    """Version and thread count, and whether this is a CPU-only build.

    The thread count matters more than it looks: torch defaults to one thread per physical
    core while numpy's own threading is set separately, so the two columns of the benchmark
    can be running on different amounts of the machine.
    """
    try:
        import torch
    except ImportError:
        return {"installed": False}
    return {
        "installed": True,
        "version": torch.__version__,
        "cpu_only_build": "+cpu" in torch.__version__ or not torch.backends.cuda.is_built(),
        "threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }


def collect() -> dict[str, Any]:
    import numpy as np

    kernel = platform.release()
    return {
        "cpu_model": _cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "memory_gib": _mem_total_gib(),
        "kernel": kernel,
        "is_wsl": "microsoft" in kernel.lower(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        # Left unset on this machine; recorded because it silently changes numpy's threading.
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch": _torch_facts(),
    }


def render(env: dict[str, Any]) -> str:
    torch = env["torch"]
    if torch["installed"]:
        build = "CPU-only build" if torch["cpu_only_build"] else "CUDA-capable build"
        torch_line = (
            f"torch {torch['version']} ({build}, {torch['threads']} threads, "
            f"cuda_available={torch['cuda_available']})"
        )
    else:
        torch_line = "torch not installed"
    wsl = " (WSL2)" if env["is_wsl"] else ""
    omp = env["omp_num_threads"] or "unset"
    return (
        f"{env['cpu_model']}, {env['cpu_logical_cores']} logical cores, "
        f"{env['memory_gib']} GiB\n"
        f"{env['platform']}{wsl}\n"
        f"python {env['python']}, numpy {env['numpy']}, OMP_NUM_THREADS={omp}\n"
        f"{torch_line}"
    )
