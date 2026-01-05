from __future__ import annotations

from typing import Dict, Optional, Sequence, Set, Tuple, List

CAP_DEFAULT = 4

Tube = Tuple[str, ...]
State = Tuple[Tube, ...]


def can_put(restricted_put: Dict[int, str], dst_idx_1based: int, color: str) -> bool:
    """목적지 병에 해당 색을 '붓는 것'이 허용되는지."""
    if dst_idx_1based in restricted_put and restricted_put[dst_idx_1based] != color:
        return False
    return True


def can_pour(
    tube_caps_1based: Sequence[int],
    restricted_put: Dict[int, str],
    locked_out: Set[int],
    no_put: Set[int],
    src: Tube,
    dst: Tube,
    src_i_1based: int,
    dst_i_1based: int,
) -> bool:
    """같은색 위 또는 빈병, + 목적지 병의 '넣기 제한' 준수. + 잠금병(고)은 꺼내기 불가."""
    if dst_i_1based in no_put:
        return False
    if src_i_1based in locked_out:
        return False
    cap_dst = tube_caps_1based[dst_i_1based]
    if not src or len(dst) == cap_dst:
        return False
    c = src[0]
    if not can_put(restricted_put, dst_i_1based, c):
        return False
    return (not dst) or (dst[0] == c)


def pour(tube_caps_1based: Sequence[int], state: State, i0: int, j0: int) -> Optional[tuple[State, int, str]]:
    """
    i0->j0 (0-based). src top의 연속 같은색을 가능한 만큼 한 번에 붓기.
    반환: (새 상태, 이동칸수, 색) 또는 None
    """
    src = list(state[i0])
    dst = list(state[j0])
    if not src:
        return None
    c = src[0]

    k = 0
    while k < len(src) and src[k] == c:
        k += 1

    cap_dst = tube_caps_1based[j0 + 1]
    space = cap_dst - len(dst)
    move = min(k, space)
    if move == 0:
        return None

    moved = src[:move]
    src = src[move:]
    dst = moved + dst

    ns = list(state)
    ns[i0] = tuple(src)
    ns[j0] = tuple(dst)
    return tuple(ns), move, c


def print_state(
    state: Sequence[Tube],
    restricted_put: Dict[int, str],
    locked_out: Set[int],
    no_put: Set[int] | None = None,
    tube_caps_1based: Sequence[int] | None = None,
) -> None:
    no_put = no_put or set()
    for idx, tube in enumerate(state, 1):
        tags = []
        if idx in restricted_put:
            tags.append(f"넣기 제한: {restricted_put[idx]}만")
        if idx in locked_out:
            tags.append("고")
        if idx in no_put:
            tags.append("재투입불가")
        if tube_caps_1based is not None and tube_caps_1based[idx] != CAP_DEFAULT:
            tags.append(f"{tube_caps_1based[idx]}칸")
        tag = f" ({', '.join(tags)})" if tags else ""
        print(f"{idx:2d}: {list(tube)}{tag}")


