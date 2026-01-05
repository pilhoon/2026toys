from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple, Iterable, Optional

import possible
try:
    import ray  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "ray가 필요합니다. 프로젝트 venv를 활성화해서 실행하세요:\n"
        "  cd /Users/euler.calc/playground/magicsort\n"
        "  . .venv/bin/activate\n"
        "  python unknown_solutions.py\n"
        f"(import ray failed: {e!r})"
    )


# 퍼즐을 이 문자열에 넣으세요. (각 줄 = 병 1개, 위->아래)
# 지원 문법:
# - compact: '주보연회' (1글자 색 연속)
# - 반복: '녹2' (2번 반복), 미지 반복: '?3' (서로 다른 미지 3개 생성)
# - 빈 병: '빈'
# - 제한: '/갈' (그 병은 갈만 다시 넣기 가능)  예) '하녹2갈/갈'
# - '고': 넣기는 가능하지만 꺼내기 불가(잠금 병) 예) '고주보연회' 또는 '주보연회 고'
# - 한칸짜리 병: 라인 끝에 '1' / '1x' (1x는 꺼내기 불가)
UNKNOWN_PUZZLE_RAW = """
갈3노
보레
레2하보
회?보
노하

노2?갈
핑2
파3
-

하2파회
-
레핑보회

""".strip()


def _collect_unknowns(state: possible.State) -> List[str]:
    toks = sorted({c for tube in state for c in tube if c.startswith("?")}, key=lambda x: int(x[1:]))
    return toks


def _collect_known_colors(state: possible.State, restricted_put: Dict[int, str]) -> List[str]:
    colors = {c for tube in state for c in tube if not c.startswith("?")}
    colors |= set(restricted_put.values())
    return sorted(colors)


def _apply_assignment(state: possible.State, mapping: Dict[str, str]) -> possible.State:
    out: List[Tuple[str, ...]] = []
    for tube in state:
        nt = []
        for c in tube:
            if c.startswith("?"):
                nt.append(mapping[c])
            else:
                nt.append(c)
        out.append(tuple(nt))
    return tuple(out)

def _solve_one(
    mapping: Dict[str, str],
    base_state: possible.State,
    restricted: Dict[int, str],
    locked_out: set[int],
    tube_caps: List[int],
    no_put: set[int],
    goal: str,
    solver: str,
    bfs_max_nodes: int,
    beam_time: float,
    beam_width: int,
    beam_steps: int,
) -> tuple[bool, Dict[str, str], Optional[List[tuple[int, int, int, str]]], Optional[possible.State], Optional[tuple[int, int, int, str]]]:
    """
    단일 mapping에 대해 해법 존재 여부를 평가.
    반환: (solved, mapping, path, final_state, first_move)
    """
    state = _apply_assignment(base_state, mapping)

    # possible.py 내부 루프가 전역 N/restricted_put을 쓰므로 갱신(프로세스별로 독립)
    possible.restricted_put = restricted
    possible.locked_out = locked_out
    possible.no_put = no_put
    possible.tube_caps = tube_caps
    possible.N = len(state)

    if solver == "bfs":
        r = possible.bfs_solve(state, goal=goal, max_nodes=bfs_max_nodes)
        if not r["solved"]:
            return False, mapping, None, None, None
        path = r["path"]
        final_state = r["state"]
    elif solver == "beam":
        r = possible.beam_solve(state, goal=goal, beam_width=beam_width, max_steps=beam_steps, time_limit_sec=beam_time)
        if not r["solved"]:
            return False, mapping, None, None, None
        path = r["path"]
        final_state = r["state"]
    else:
        raise ValueError(f"Unknown SOLVER={solver!r} (use bfs|beam)")

    first_move = tuple(path[0]) if path else None
    return True, mapping, path, final_state, first_move


def _strict_count_ok(state: possible.State) -> bool:
    # 엄격 완료(4칸 단색/빈 병)가 가능하려면 각 색 개수는 0 또는 CAP이어야 함.
    counts = possible.color_counts(state)
    return all(n in (0, possible.CAP) for n in counts.values())


def _iter_assignments(
    unknowns: List[str],
    colors: List[str],
    distinct: bool,
    max_assignments: Optional[int],
) -> Iterable[Dict[str, str]]:
    if not unknowns:
        yield {}
        return

    # legacy: no pruning, full cartesian/permutation via backtracking
    n = 0
    used: set[str] = set()
    mapping: Dict[str, str] = {}

    def rec(i: int) -> Iterable[Dict[str, str]]:
        nonlocal n
        if max_assignments is not None and n >= max_assignments:
            return
        if i == len(unknowns):
            n += 1
            yield dict(mapping)
            return
        u = unknowns[i]
        for c in colors:
            if distinct and c in used:
                continue
            mapping[u] = c
            if distinct:
                used.add(c)
            yield from rec(i + 1)
            if distinct:
                used.remove(c)
            del mapping[u]

    yield from rec(0)


def _iter_assignments_with_color_totals(
    unknowns: List[str],
    colors: List[str],
    distinct: bool,
    known_counts: Dict[str, int],
    color_total: int,
    max_assignments: Optional[int],
) -> Iterable[Dict[str, str]]:
    """
    색 총개수 제약(각 색은 최대 color_total개)을 이용한 프루닝.
    - known_counts는 미지('?') 제외한 현재 보이는 색 개수.
    - 생성 단계에서 이미 total을 초과하는 조합은 만들지 않음.
    """
    if not unknowns:
        yield {}
        return
    if color_total <= 0:
        yield from _iter_assignments(unknowns, colors, distinct=distinct, max_assignments=max_assignments)
        return

    remaining: Dict[str, int] = {}
    for c in colors:
        rem = color_total - known_counts.get(c, 0)
        if rem > 0:
            remaining[c] = rem

    if distinct:
        # distinct일 때는 각 색을 1번만 쓸 수 있으니, 최소 1개라도 남아있는 색만
        remaining = {c: 1 for c, rem in remaining.items() if rem > 0}

    if len(remaining) == 0:
        return

    n = 0
    mapping: Dict[str, str] = {}

    def rec(i: int) -> Iterable[Dict[str, str]]:
        nonlocal n
        if max_assignments is not None and n >= max_assignments:
            return
        if i == len(unknowns):
            n += 1
            yield dict(mapping)
            return
        u = unknowns[i]
        for c, rem in remaining.items():
            if rem <= 0:
                continue
            remaining[c] = rem - 1
            mapping[u] = c
            yield from rec(i + 1)
            del mapping[u]
            remaining[c] = rem

    yield from rec(0)


def _iter_assignments_full_exact(
    unknowns: List[str],
    known_counts: Dict[str, int],
    colors: List[str],
    color_total: int,
    max_assignments: Optional[int],
) -> Iterable[Dict[str, str]]:
    """
    full(전체문제, 각 색 정확히 color_total개)인 경우:
    미지들이 채워야 하는 '정확한 멀티셋'이 계산 가능하면 그것만 대입한다.

    조건:
      sum(max(0, color_total - known_counts[c])) == len(unknowns)
    """
    needed: Dict[str, int] = {}
    for c in colors:
        rem = color_total - known_counts.get(c, 0)
        if rem > 0:
            needed[c] = rem

    if sum(needed.values()) != len(unknowns):
        return

    n = 0
    mapping: Dict[str, str] = {}

    def rec(i: int) -> Iterable[Dict[str, str]]:
        nonlocal n
        if max_assignments is not None and n >= max_assignments:
            return
        if i == len(unknowns):
            n += 1
            yield dict(mapping)
            return
        u = unknowns[i]
        for c in list(needed.keys()):
            if needed[c] <= 0:
                continue
            needed[c] -= 1
            mapping[u] = c
            yield from rec(i + 1)
            del mapping[u]
            needed[c] += 1

    yield from rec(0)


def main() -> None:
    # 설정 (환경변수로 조정)
    goal = os.getenv("GOAL", "relaxed").strip()  # relaxed | strict
    distinct = os.getenv("UNKNOWN_DISTINCT", "1").strip() != "0"
    solver = os.getenv("SOLVER", "bfs").strip().lower()  # bfs | beam
    bfs_max_nodes = int(os.getenv("BFS_MAX_NODES", "500000"))
    beam_time = float(os.getenv("BEAM_TIME", "5.0"))
    beam_width = int(os.getenv("BEAM_WIDTH", "8000"))
    beam_steps = int(os.getenv("BEAM_STEPS", "350"))
    max_assignments = os.getenv("MAX_ASSIGNMENTS")
    max_assignments_n = int(max_assignments) if max_assignments else None
    # 색 총개수 제약(프루닝용). 기본 4. 0이면 비활성.
    color_total = int(os.getenv("COLOR_TOTAL", "4"))
    # 병렬 처리 설정
    max_procs = min(int(os.getenv("MAX_PROCS", "8")), os.cpu_count() or 1)

    # 퍼즐 파싱
    base_state, restricted, locked_out, tube_caps, no_put, mode = possible.parse_initial_state(UNKNOWN_PUZZLE_RAW, cap=possible.CAP)
    unknowns = _collect_unknowns(base_state)
    colors = _collect_known_colors(base_state, restricted)
    known_counts = possible.color_counts(base_state)

    print("=== UNKNOWN SOLVER ===")
    print(f"goal={goal}, solver={solver}, distinct_unknowns={distinct}")
    print(f"unknowns({len(unknowns)}): {unknowns}")
    print(f"known_colors({len(colors)}): {colors}")
    print(f"restricted_put: {restricted}")
    print(f"locked_out: {sorted(locked_out)}")
    print(f"no_put: {sorted(no_put)}")
    print(f"COLOR_TOTAL_MODE(from header/env): {mode}")
    print(f"COLOR_TOTAL={color_total} (0이면 프루닝 비활성)")
    if color_total > 0:
        feasible_colors = [c for c in colors if known_counts.get(c, 0) < color_total]
        print(f"candidate_colors_by_total({len(feasible_colors)}): {feasible_colors}")
        if mode == "full":
            missing_multiset: List[str] = []
            for c in feasible_colors:
                missing_multiset.extend([c] * (color_total - known_counts.get(c, 0)))
            print(f"missing_color_multiset({len(missing_multiset)}): {missing_multiset}")
            if distinct and len(set(missing_multiset)) != len(missing_multiset):
                print("NOTE: full 모드에서 필요한 멀티셋에 중복이 있어 UNKNOWN_DISTINCT=1이면 조합이 0개가 됩니다.")
    print("\n=== base state ===")
    possible.restricted_put = restricted
    possible.locked_out = locked_out
    possible.no_put = no_put
    possible.tube_caps = tube_caps
    possible.N = len(base_state)
    possible.print_state(base_state)

    # 조합 시도
    total = 0
    solved = 0
    first_move_possible: set[tuple[int, int, int, str]] = set()
    if mode == "full" and color_total > 0:
        # full 모드에서는 '정확히 4개'를 맞추는 멀티셋이 계산 가능하면 그것만 대입(가장 강력한 프루닝)
        assignments = _iter_assignments_full_exact(
            unknowns,
            known_counts=known_counts,
            colors=colors,
            color_total=color_total,
            max_assignments=max_assignments_n,
        )
    else:
        assignments = _iter_assignments_with_color_totals(
            unknowns,
            colors,
            distinct=distinct,
            known_counts=known_counts,
            color_total=color_total,
            max_assignments=max_assignments_n,
        )
    # Ray 전용 실행 (최대 8 프로세스)
    ray.init(num_cpus=max_procs, ignore_reinit_error=True, include_dashboard=False)

    @ray.remote
    def solve_remote(mapping: Dict[str, str]):
        return _solve_one(
            mapping,
            base_state,
            restricted,
            locked_out,
            tube_caps,
            no_put,
            goal,
            solver,
            bfs_max_nodes,
            beam_time,
            beam_width,
            beam_steps,
        )

    max_inflight = max_procs * 8
    inflight = []
    it = iter(assignments)
    done_iter = False

    while True:
        while not done_iter and len(inflight) < max_inflight:
            try:
                mapping = next(it)
            except StopIteration:
                done_iter = True
                break
            # strict 모드 early prune
            if goal == "strict":
                st = _apply_assignment(base_state, mapping)
                if not _strict_count_ok(st):
                    continue
            total += 1
            inflight.append(solve_remote.remote(mapping))

        if not inflight:
            break

        ready, inflight = ray.wait(inflight, num_returns=1)
        ok, mapping, path, final_state, first_move = ray.get(ready[0])
        if not ok:
            continue
        solved += 1
        if first_move is not None:
            first_move_possible.add(first_move)
        print("\n========================")
        print(f"SOLUTION #{solved}  (assignment {total})")
        print("mapping:", mapping)
        print("moves:", len(path or []))
        if path:
            possible.print_path(path)
        print("\nfinal state:")
        possible.print_state(final_state)  # type: ignore[arg-type]

    ray.shutdown()

    print("\n=== DONE ===")
    print(f"tried_assignments={total}, solved={solved}")
    if not first_move_possible:
        print("first_move_possible_set: (no solved paths with at least 1 move)")
    else:
        # Each move is (from, to, amount, color)
        print(f"first_move_possible_set({len(first_move_possible)}): {first_move_possible}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # when piping to head/tail, stdout may close early
        pass


