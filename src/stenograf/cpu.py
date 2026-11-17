"""Where background inference runs: on the efficiency cores, below everything
interactive, and never spin-waiting.

One rule on every platform — as many threads as the CPU has efficiency cores,
confined to them (every physical core, unconfined, when it has only one kind)
— reached through the one mechanism each OS offers. macOS is the exception in
the count, not the rule: it has no CPU affinity, background QoS on the
*calling* thread is the only way onto the E-cores, and ONNX Runtime's pool
threads never take it on, wherever the session was created — so macOS runs
one thread. Linux and Windows pin, because Linux at least places a lone
background thread on a fast core even at ``SCHED_IDLE``. The measurements
behind each choice (M4 Max and a Ryzen AI 9 HX 370; Windows unmeasured) are
in ``eval/README.md``, "Background threads, by platform".
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

EFFICIENT_SHARE = 0.85
"""A core is an efficiency core when its peak capacity is at most this share
of the fastest core's. Fast cores differ among themselves by a few percent
(preferred-core boost); efficiency cores sit far below — the HX 370's Zen5c
peak at 64 % of its Zen5, Intel's E-cores near three quarters of their
P-cores."""

MAX_THREADS = 16
"""Ceiling on a background session's threads: no CPU a meeting runs on has
more efficiency cores, and a many-core server gains nothing a 27-s chunk
step can use."""

_FALLBACK_THREADS = 4


@dataclass(frozen=True)
class Cpu:
    """One logical processor this process may run on."""

    id: int
    """0-based logical processor number, as the OS's affinity calls count."""
    core: tuple[int, int]
    """The physical core it belongs to (package, core): SMT siblings share it."""
    capacity: int
    """Peak capacity in any unit the platform reports; only ratios matter."""


@dataclass(frozen=True)
class ThreadPlan:
    """How many threads a background session runs, and where."""

    threads: int
    cpus: frozenset[int] | None
    """Logical processors to confine every thread to; ``None`` = anywhere."""


def plan_for(cpus: list[Cpu]) -> ThreadPlan:
    """The rule, on a known set of processors."""
    if not cpus:
        return ThreadPlan(_FALLBACK_THREADS, None)
    top = max(cpu.capacity for cpu in cpus)
    efficient = [cpu for cpu in cpus if cpu.capacity <= EFFICIENT_SHARE * top]
    chosen = efficient or cpus
    threads = max(1, min(len({cpu.core for cpu in chosen}), MAX_THREADS))
    return ThreadPlan(threads, frozenset(cpu.id for cpu in chosen) if efficient else None)


@functools.cache
def background_plan() -> ThreadPlan:
    """This machine's plan, read once. Never raises: a platform that will not
    say gets a few unconfined threads."""
    if sys.platform == "darwin":
        return ThreadPlan(1, None)
    try:
        if sys.platform.startswith("linux"):
            return plan_for(_linux_cpus())
        if sys.platform == "win32":
            return plan_for(_windows_cpus())
    except Exception:  # noqa: BLE001 — an unreadable topology only loses the pinning
        pass
    return ThreadPlan(_FALLBACK_THREADS, None)


def enter_background(plan: ThreadPlan) -> None:
    """Lower the calling thread below everything interactive and confine it to
    ``plan.cpus`` — best effort, a no-op where the platform refuses.

    macOS: background QoS, which keeps the thread on the efficiency cores.
    Linux: ``SCHED_IDLE``, which runs it only on otherwise idle CPU time.
    Windows: background mode (lowest CPU, I/O and memory priority)."""
    with contextlib.suppress(Exception):
        if sys.platform == "darwin":
            libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
            libc.pthread_set_qos_class_self_np(0x09, 0)  # QOS_CLASS_BACKGROUND
        elif sys.platform.startswith("linux"):
            os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))  # 0 = this thread
            if plan.cpus:
                os.sched_setaffinity(0, plan.cpus)
        elif sys.platform == "win32":
            kernel32 = _kernel32()
            thread = kernel32.GetCurrentThread()
            kernel32.SetThreadPriority(thread, 0x00010000)  # THREAD_MODE_BACKGROUND_BEGIN
            if plan.cpus:
                kernel32.SetThreadAffinityMask(thread, sum(1 << cpu for cpu in plan.cpus))


def onnx_affinities(plan: ThreadPlan) -> str | None:
    """``session.intra_op_thread_affinities`` for ``plan``: one group per pool
    thread (the calling thread is the session's first and is pinned by
    :func:`enter_background`), processors counted from 1 as ONNX Runtime
    expects. ``None`` when there is nothing to confine."""
    if not plan.cpus or plan.threads < 2:
        return None
    group = ",".join(str(cpu + 1) for cpu in sorted(plan.cpus))
    return ";".join([group] * (plan.threads - 1))


def _linux_cpus() -> list[Cpu]:
    base = Path("/sys/devices/system/cpu")
    ids = sorted(os.sched_getaffinity(0))  # pyright: ignore[reportAttributeAccessIssue]

    def read(i: int, name: str) -> int | None:
        path = base / f"cpu{i}" / name
        return int(path.read_text()) if path.exists() else None

    # cpu_capacity is the scheduler's own ranking where it ranks (ARM
    # big.LITTLE); x86 kernels report a flat 1024 for every core even on a
    # hybrid CPU (Ryzen AI 9 HX 370, 2026-11-17), so a flat ranking defers to
    # the peak frequency.
    capacity = {i: read(i, "cpu_capacity") for i in ids}
    if None in capacity.values() or len(set(capacity.values())) < 2:
        capacity = {i: read(i, "cpufreq/cpuinfo_max_freq") for i in ids}
    cpus = []
    for i in ids:
        package, core = read(i, "topology/physical_package_id"), read(i, "topology/core_id")
        place = (0 if package is None else package, i if core is None else core)
        cpus.append(Cpu(i, place, capacity[i] or 1))  # unreadable or 0: one kind of core
    return cpus


def _kernel32():
    kernel32 = ctypes.WinDLL("kernel32")  # pyright: ignore[reportAttributeAccessIssue]
    kernel32.GetCurrentThread.restype = ctypes.c_void_p
    kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
    kernel32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel32.SetThreadAffinityMask.restype = ctypes.c_size_t
    return kernel32


def _windows_cpus() -> list[Cpu]:
    kernel32 = ctypes.WinDLL("kernel32")  # pyright: ignore[reportAttributeAccessIssue]
    length = ctypes.c_ulong(0)
    kernel32.GetSystemCpuSetInformation(None, 0, ctypes.byref(length), None, 0)
    buffer = ctypes.create_string_buffer(length.value)
    if not kernel32.GetSystemCpuSetInformation(buffer, length, ctypes.byref(length), None, 0):
        raise OSError("GetSystemCpuSetInformation failed")
    return parse_cpu_sets(buffer.raw[: length.value])


def parse_cpu_sets(raw: bytes) -> list[Cpu]:
    """Processor group 0 of a ``SYSTEM_CPU_SET_INFORMATION`` array.

    Each record: Size and Type (DWORDs), then for a CPU set Id (DWORD), Group
    (WORD), LogicalProcessorIndex, CoreIndex, LastLevelCacheIndex,
    NumaNodeIndex and EfficiencyClass (BYTEs). EfficiencyClass rises with
    performance, 0 being the most efficient. Only group 0 is read: a thread's
    affinity mask covers one group."""
    cpus = []
    offset = 0
    while offset + 8 <= len(raw):
        size, kind = struct.unpack_from("<II", raw, offset)
        if size == 0:
            break
        if kind == 0 and size >= 19:  # CpuSetInformation
            group, logical, core, _cache, _numa, efficiency = struct.unpack_from(
                "<HBBBBB", raw, offset + 12
            )
            if group == 0:
                cpus.append(Cpu(logical, (0, core), efficiency + 1))
        offset += size
    return cpus
