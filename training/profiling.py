"""Host stage timings and CUDA activity for a short sample of training."""

import json
from contextlib import nullcontext
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter


_NO_REGION = nullcontext()
_PREFIX = "sokoban/"
_STAGES = {
    "environment": "推进环境",
    "data_prep": "准备数据",
    "transfer": "传输与取回结果",
    "policy_compute": "策略推理",
    "train_compute": "网络更新",
    "action_selection": "选择动作",
    "other": "其余主机时间",
}


def null_region(name: str):
    return _NO_REGION


def _active_ms(intervals: list[tuple[float, float]]) -> float:
    """Union of microsecond intervals, including work on overlapping CUDA streams."""
    covered_us = 0.0
    end_us = float("-inf")
    for start, end in sorted(intervals):
        covered_us += max(0.0, end - max(start, end_us))
        end_us = max(end_us, end)
    return covered_us / 1000


def summarize_trace(events: list[dict], wall_ms: float, *, cuda: bool) -> dict:
    host = dict.fromkeys(_STAGES, 0.0)
    device_events = {"kernel": [], "gpu_memcpy": [], "gpu_memset": []}
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        if category == "user_annotation" or event.get("pid") == "CPU functions":
            stage = event["name"].removeprefix(_PREFIX)
            if event["name"].startswith(_PREFIX) and stage in host:
                host[stage] += event["dur"] / 1000
        elif cuda and category in device_events:
            start = event["ts"]
            device_events[category].append((start, start + event["dur"]))
    # Regions on the collector thread are disjoint. Other includes the final device wait.
    host["other"] = max(0.0, wall_ms - sum(host.values()))
    gpu = None
    if any(device_events.values()):
        active = _active_ms([span for group in device_events.values() for span in group])
        gpu = {
            "kernel_ms": _active_ms(device_events["kernel"]),
            "memcpy_ms": _active_ms(device_events["gpu_memcpy"]),
            "memset_ms": _active_ms(device_events["gpu_memset"]),
            "active_ms": active, "active_pct": 100 * active / wall_ms,
        }
    return {"host_ms": host, "gpu": gpu}


class TrainingProfile:
    def __init__(self, device):
        import torch

        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self._sync = torch.cuda.synchronize
        self._record = torch.profiler.record_function
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.cuda and torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(activities=activities)

    def start(self):
        if self.cuda:
            self._sync(self.device)
        self.profiler.start()
        self.started = perf_counter()

    def region(self, name: str):
        return self._record(_PREFIX + name)

    def abort(self):
        self.profiler.stop()

    def stop(self, output: Path, *, start_step: int, end_step: int, rounds: int, updates: int,
             epsilon_start: float, epsilon_end: float) -> Path:
        try:
            if self.cuda:
                self._sync(self.device)
            wall_ms = (perf_counter() - self.started) * 1000
        finally:
            self.profiler.stop()
        profiles = output / "profiles"
        profiles.mkdir(exist_ok=True)
        directory = Path(mkdtemp(prefix=f"steps-{start_step}-{end_step}-", dir=profiles))
        trace_path = directory / "trace.json"
        self.profiler.export_chrome_trace(str(trace_path))
        with trace_path.open(encoding="utf-8") as stream:
            trace = json.load(stream)
        # The CPU-only fallback exporter writes a bare event list.
        events = trace["traceEvents"] if isinstance(trace, dict) else trace
        timings = summarize_trace(events, wall_ms, cuda=self.cuda)
        summary = {
            "device": str(self.device), "start_step": start_step, "end_step": end_step,
            "rounds": rounds, "updates": updates, "epsilon_start": epsilon_start, "epsilon_end": epsilon_end,
            "wall_ms": wall_ms, **timings,
        }
        path = directory / "summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"性能采样 {start_step}→{end_step} 环境步，{updates} 次更新，实耗 {wall_ms:.1f} ms")
        for name, elapsed in sorted(timings["host_ms"].items(), key=lambda pair: -pair[1]):
            print(f"  {_STAGES[name]}  {elapsed:.1f} ms  ({elapsed / wall_ms:.1%})")
        gpu = timings["gpu"]
        if gpu is not None:
            print(f"GPU 核函数 {gpu['kernel_ms']:.1f} ms，复制 {gpu['memcpy_ms']:.1f} ms，"
                  f"置值 {gpu['memset_ms']:.1f} ms，设备活动占采样窗口 {gpu['active_pct']:.1f}%")
        elif self.cuda:
            print("此窗口未捕获 CUDA 设备活动，GPU 耗时不可用。")
        print(f"性能报告 {path}", flush=True)
        return path
