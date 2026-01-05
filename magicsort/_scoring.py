from __future__ import annotations

from typing import Dict, Sequence

import _engine

Tube = _engine.Tube
State = _engine.State


def tube_color_counts(tube: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for x in tube:
        counts[x] = counts.get(x, 0) + 1
    return counts


def tube_cohesion(tube: Sequence[str]) -> int:
    """병 안에서 가장 많은 색의 개수(빈 병이면 0)."""
    if not tube:
        return 0
    cnt = tube_color_counts(tube)
    return max(cnt.values())


def count_mono_and_complete(
    state: Sequence[Tube],
    tube_caps_1based: Sequence[int] | None = None,
) -> tuple[int, int]:
    """
    mono: 비었거나 단색인 병 개수
    complete: '용량(cap)만큼 꽉 찬' 단색 병 개수
    """
    mono = 0
    complete = 0
    for idx, t in enumerate(state, 1):
        if not t:
            mono += 1
            continue
        if len(set(t)) == 1:
            mono += 1
            cap = tube_caps_1based[idx] if tube_caps_1based is not None else _engine.CAP_DEFAULT
            if len(t) == cap:
                complete += 1
    return mono, complete


def cohesion_sum(state: Sequence[Tube]) -> int:
    """각 병의 cohesion 합."""
    return sum(tube_cohesion(t) for t in state)


def score_relaxed(state: Sequence[Tube], tube_caps_1based: Sequence[int] | None = None) -> tuple[int, int, int]:
    """
    (possible.py에서 쓰는) beam search 점수:
      1) mono(빈/단색) 병 개수 (maximize)
      2) complete(꽉 찬 단색) 병 개수 (maximize)
      3) cohesion 합 (maximize)
    """
    mono, complete = count_mono_and_complete(state, tube_caps_1based=tube_caps_1based)
    return (mono, complete, cohesion_sum(state))


def score_maximize(state: Sequence[Tube], tube_caps_1based: Sequence[int] | None = None) -> tuple[int, int, int]:
    """
    (maximize.py에서 쓰는) 점수:
      1) complete(꽉 찬 단색) 병 개수 (maximize)
      2) mono(빈/단색) 병 개수 (maximize)
      3) cohesion 합 (maximize)
    """
    mono, complete = count_mono_and_complete(state, tube_caps_1based=tube_caps_1based)
    return (complete, mono, cohesion_sum(state))


