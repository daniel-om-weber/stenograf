import struct

from stenograf import cpu
from stenograf.cpu import Cpu, ThreadPlan, onnx_affinities, parse_cpu_sets, plan_for


def _hx370() -> list[Cpu]:
    """Ryzen AI 9 HX 370 as Linux reports it: 4 Zen5 cores (logical 0-3 and
    their SMT siblings 12-15) at 5.16 GHz, 8 Zen5c cores (4-11, 16-23) at 3.29."""
    cpus = []
    for logical in range(24):
        core = logical % 12
        fast = core < 4
        cpus.append(Cpu(logical, (0, core), 5157895 if fast else 3289474))
    return cpus


def test_hybrid_cpu_runs_one_thread_per_efficiency_core_confined_there():
    plan = plan_for(_hx370())
    assert plan.threads == 8
    assert plan.cpus == frozenset([*range(4, 12), *range(16, 24)])


def test_fast_cores_differing_by_boost_are_not_efficiency_cores():
    # Preferred-core boost spreads one kind of core over a few percent.
    cpus = [Cpu(i, (0, i), 5_700_000 if i < 2 else 5_400_000) for i in range(8)]
    assert plan_for(cpus) == ThreadPlan(8, None)


def test_uniform_cpu_runs_every_physical_core_unconfined():
    cpus = [Cpu(i, (0, i // 2), 4_000_000) for i in range(16)]  # 8 cores, SMT
    assert plan_for(cpus) == ThreadPlan(8, None)


def test_thread_count_is_capped():
    cpus = [Cpu(i, (0, i), 1) for i in range(64)]
    assert plan_for(cpus).threads == cpu.MAX_THREADS


def test_unknown_topology_falls_back_unconfined():
    plan = plan_for([])
    assert plan.cpus is None and plan.threads >= 1


def test_onnx_affinities_name_the_cores_from_one_for_every_pool_thread():
    plan = ThreadPlan(3, frozenset({4, 5}))
    assert onnx_affinities(plan) == "5,6;5,6"  # the calling thread is pinned separately
    assert onnx_affinities(ThreadPlan(1, frozenset({4}))) is None
    assert onnx_affinities(ThreadPlan(8, None)) is None


def _cpu_set(logical: int, core: int, efficiency: int, group: int = 0) -> bytes:
    # SYSTEM_CPU_SET_INFORMATION, CpuSetInformation: Size, Type, Id, Group,
    # LogicalProcessorIndex, CoreIndex, LastLevelCacheIndex, NumaNodeIndex,
    # EfficiencyClass, AllFlags, Reserved, AllocationTag — 32 bytes.
    return struct.pack(
        "<IIIHBBBBBBIQ", 32, 0, 256 + logical, group, logical, core, 0, 0, efficiency, 0, 0, 0
    )


def test_windows_cpu_sets_parse_into_efficiency_classes():
    # 2 P-cores with SMT (class 1), 4 E-cores (class 0), one processor in group 1.
    raw = b"".join(
        [_cpu_set(0, 0, 1), _cpu_set(1, 0, 1), _cpu_set(2, 2, 1), _cpu_set(3, 2, 1)]
        + [_cpu_set(4 + i, 4 + i, 0) for i in range(4)]
        + [_cpu_set(0, 0, 1, group=1)]
    )
    cpus = parse_cpu_sets(raw)
    assert len(cpus) == 8
    assert plan_for(cpus) == ThreadPlan(4, frozenset({4, 5, 6, 7}))


def test_windows_cpu_sets_of_one_class_confine_nothing():
    raw = b"".join(_cpu_set(i, i // 2, 0) for i in range(8))
    assert plan_for(parse_cpu_sets(raw)) == ThreadPlan(4, None)


def test_macos_runs_one_unconfined_thread(monkeypatch):
    monkeypatch.setattr(cpu.sys, "platform", "darwin")
    cpu.background_plan.cache_clear()
    try:
        assert cpu.background_plan() == ThreadPlan(1, None)
    finally:
        cpu.background_plan.cache_clear()


def test_unreadable_topology_never_raises(monkeypatch):
    monkeypatch.setattr(cpu.sys, "platform", "linux")

    def broken() -> list[Cpu]:
        raise OSError("no sysfs")

    monkeypatch.setattr(cpu, "_linux_cpus", broken)
    cpu.background_plan.cache_clear()
    try:
        plan = cpu.background_plan()
    finally:
        cpu.background_plan.cache_clear()
    assert plan.cpus is None and plan.threads >= 1


def _fake_sysfs(tmp_path, cpus: dict[int, dict[str, int]]):
    for i, files in cpus.items():
        for name, value in files.items():
            path = tmp_path / f"cpu{i}" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{value}\n")


def _linux_cpus_from(tmp_path, monkeypatch):
    real = cpu.Path
    monkeypatch.setattr(
        cpu, "Path", lambda p: tmp_path if p == "/sys/devices/system/cpu" else real(p)
    )
    monkeypatch.setattr(cpu.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    return cpu._linux_cpus()


def test_linux_flat_capacity_defers_to_peak_frequency(tmp_path, monkeypatch):
    # x86 kernels write 1024 for every core even on a hybrid CPU.
    _fake_sysfs(
        tmp_path,
        {
            i: {
                "cpu_capacity": 1024,
                "cpufreq/cpuinfo_max_freq": 5_000_000 if i < 2 else 3_000_000,
                "topology/physical_package_id": 0,
                "topology/core_id": i,
            }
            for i in range(4)
        },
    )
    assert plan_for(_linux_cpus_from(tmp_path, monkeypatch)) == ThreadPlan(2, frozenset({2, 3}))


def test_linux_ranked_capacity_wins_over_frequency(tmp_path, monkeypatch):
    # ARM big.LITTLE: the scheduler's ranking, even where frequencies say otherwise.
    _fake_sysfs(
        tmp_path,
        {
            i: {
                "cpu_capacity": 1024 if i == 0 else 400,
                "cpufreq/cpuinfo_max_freq": 2_000_000,
                "topology/physical_package_id": 0,
                "topology/core_id": i,
            }
            for i in range(4)
        },
    )
    assert plan_for(_linux_cpus_from(tmp_path, monkeypatch)) == ThreadPlan(3, frozenset({1, 2, 3}))


def test_linux_smt_siblings_of_core_zero_count_once(tmp_path, monkeypatch):
    # Core 0's sibling must not count as a core of its own (core_id 0 is falsy).
    _fake_sysfs(
        tmp_path,
        {
            i: {
                "cpufreq/cpuinfo_max_freq": 4_000_000,
                "topology/physical_package_id": 0,
                "topology/core_id": i % 2,
            }
            for i in range(4)
        },
    )
    assert plan_for(_linux_cpus_from(tmp_path, monkeypatch)) == ThreadPlan(2, None)


def test_session_options_never_spin_and_carry_the_plan():
    bare = cpu.session_options(None)
    assert bare.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    assert bare.intra_op_num_threads == 0  # ONNX Runtime's own count
    pinned = cpu.session_options(ThreadPlan(3, frozenset({4, 5})))
    assert pinned.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    assert pinned.intra_op_num_threads == 3
    assert pinned.get_session_config_entry("session.intra_op_thread_affinities") == "5,6;5,6"


def test_live_captions_keep_onnx_runtimes_count_on_macos(monkeypatch):
    monkeypatch.setattr(cpu.sys, "platform", "darwin")
    assert cpu.live_plan() is None
    monkeypatch.setattr(cpu.sys, "platform", "linux")
    monkeypatch.setattr(cpu, "background_plan", lambda: ThreadPlan(8, frozenset({4})))
    assert cpu.live_plan() == ThreadPlan(8, frozenset({4}))


def _fake_affinity(monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(cpu.sys, "platform", "linux")
    monkeypatch.setattr(cpu.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    monkeypatch.setattr(
        cpu.os, "sched_setaffinity", lambda pid, cpus: calls.append(set(cpus)), raising=False
    )
    return calls


def test_confined_puts_the_thread_back_afterwards(monkeypatch):
    calls = _fake_affinity(monkeypatch)
    try:
        with cpu.confined(ThreadPlan(2, frozenset({2, 3}))):
            assert calls == [{2, 3}]
            raise ValueError("decode failed")
    except ValueError:
        pass
    assert calls == [{2, 3}, {0, 1, 2, 3}]


def test_confined_without_cores_touches_nothing(monkeypatch):
    calls = _fake_affinity(monkeypatch)
    with cpu.confined(None), cpu.confined(ThreadPlan(8, None)):
        pass
    assert calls == []
