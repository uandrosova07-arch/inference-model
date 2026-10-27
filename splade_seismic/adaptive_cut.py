"""Адаптивный cut для SEISMIC: наименьшее m, при котором top-m координат запроса несут долю gamma его L1-массы.

Вместо фиксированного --query-cut передавайте в поиск gamma_cut(...) для каждого запроса.
"""
import numpy as np


def gamma_cut(values, gamma: float, cmin: int = 1, cmax: int | None = None) -> int:
    """values — ненулевые веса запроса (в любом порядке); результат ограничен [cmin, cmax]."""
    v = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    c = np.cumsum(v) / v.sum()
    m = int(np.searchsorted(c, gamma, side="left")) + 1  # первый индекс, где c >= gamma
    hi = len(v) if cmax is None else min(cmax, len(v))
    return max(cmin, min(m, hi))


def top_entries(indices, values, cut: int):
    """Первые cut координат запроса по убыванию веса: (индексы, значения)."""
    order = np.argsort(-np.asarray(values))[:cut]
    return np.asarray(indices)[order], np.asarray(values)[order]
