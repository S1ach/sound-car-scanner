# profiler.py
"""
Замер скорости блоков в реальном времени.

    @profile_real_time(budget_ms=3.0)
    def heavy_step(...): ...

    with measure("pipeline:frame", budget_ms=20.0):
        ...

    print(report())
"""
import functools
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional


class RealTimeBudgetWarning(RuntimeWarning):
    pass


@dataclass
class TimingStats:
    name: str
    budget_ms: Optional[float] = None
    calls: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    last_ms: float = 0.0
    overruns: int = 0

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


_STATS: Dict[str, TimingStats] = {}
# Предупреждать о превышении бюджета не чаще, чем раз в N превышений одного блока
WARN_EVERY = 100


def _record(name: str, budget_ms: Optional[float], elapsed_ms: float) -> None:
    stats = _STATS.get(name)
    if stats is None:
        stats = _STATS[name] = TimingStats(name=name, budget_ms=budget_ms)
    stats.calls += 1
    stats.total_ms += elapsed_ms
    stats.last_ms = elapsed_ms
    stats.max_ms = max(stats.max_ms, elapsed_ms)
    if budget_ms is not None and elapsed_ms > budget_ms:
        stats.overruns += 1
        if stats.overruns % WARN_EVERY == 1:
            warnings.warn(
                f"[{name}] {elapsed_ms:.2f} мс > бюджета {budget_ms:.2f} мс (превышений: {stats.overruns})",
                RealTimeBudgetWarning,
                stacklevel=3,
            )


@contextmanager
def measure(name: str, budget_ms: Optional[float] = None) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        _record(name, budget_ms, (time.perf_counter() - start) * 1000.0)


def profile_real_time(_func: Optional[Callable] = None, *, name: Optional[str] = None,
                      budget_ms: Optional[float] = None):
    """Декоратор: можно использовать как @profile_real_time и как @profile_real_time(budget_ms=5)."""

    def decorator(func: Callable) -> Callable:
        label = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with measure(label, budget_ms):
                return func(*args, **kwargs)

        return wrapper

    return decorator(_func) if _func is not None else decorator


def get_stats() -> Dict[str, TimingStats]:
    return dict(_STATS)


def reset_stats() -> None:
    _STATS.clear()


def report() -> str:
    if not _STATS:
        return "Профайлер: нет данных"
    header = f"{'блок':<34}{'вызовов':>9}{'сред, мс':>10}{'макс, мс':>10}{'бюджет':>9}{'превыш.':>9}"
    lines = [header, "-" * len(header)]
    for s in sorted(_STATS.values(), key=lambda s: s.name):
        budget = f"{s.budget_ms:.1f}" if s.budget_ms is not None else "-"
        lines.append(f"{s.name:<34}{s.calls:>9}{s.mean_ms:>10.3f}{s.max_ms:>10.3f}{budget:>9}{s.overruns:>9}")
    return "\n".join(lines)
