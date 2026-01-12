"""
물 정렬 퍼즐 솔버

규칙:
1. 각 병이 [비어있음] 또는 [4칸 단색]이면 승리
2. 붓기: 같은 색 위 또는 빈 병에만 가능, 연속 같은 색은 한 번에 이동
3. 미지 블록(?)이 맨 위로 드러나면 정지 (해당 병 사용 불가)

점수 기준 (우선순위 순):
1. 전체 청크 수 → 적을수록 좋음
2. 빈 병 개수 → 많을수록 좋음

최적 상태: 청크 수 = 색의 개수
"""

from __future__ import annotations
import heapq
import os
import re
import time
from typing import Dict, List, Optional, Set, Tuple

CAP = 4

State = Tuple[Tuple[str, ...], ...]

# ============================================================
# 파싱
# ============================================================

def parse_initial_state(raw: str, cap: int = CAP) -> Tuple[State, Dict[int, str], Set[int], List[int], str]:
    """
    초기 상태 파싱.
    
    반환: (state, restricted_put, locked_out, tube_caps, mode)
    mode: "full" 또는 "partial" ("(부분)" 혹은 "partial" 헤더로 설정)
    """
    if not raw.strip():
        raise ValueError("initial state text is empty")

    lines: List[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if s:
            lines.append(s)

    # 모드 헤더 처리
    mode = "full"
    if lines:
        header = lines[0].strip().lower()
        if header in ("(부분)", "partial"):
            mode = "partial"
            lines = lines[1:]

    restricted: Dict[int, str] = {}
    tubes: List[Tuple[str, ...]] = []
    locked: Set[int] = set()
    caps_1based: List[int] = [0]
    next_q = 1

    # 영어 한글자 → 한글 색 매핑
    eng_to_kor = {
        "r": "적",
        "o": "주",
        "y": "노",
        "l": "연",
        "s": "하",
        "b": "파",
        "v": "보",
        "p": "핑",
        "g": "녹",
        "e": "회",
        "w": "갈",
        "m": "자",
    }

    def normalize_color(tok: str) -> str:
        # 영어 한글자 변환
        if tok.lower() in eng_to_kor:
            tok = eng_to_kor[tok.lower()]
        # 한글 별칭 처리
        if tok in ("레", "적"):
            return "빨"
        if tok == "청":
            return "파"
        return tok

    def tokenize_base(base: str) -> List[str]:
        s = "".join(base.split())
        if not s or s == "빈":
            return []
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) == 1 and parts[0] == "빈":
                return []
            return parts

        out: List[str] = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "?":
                j = i + 1
                if j < len(s) and s[j].isdigit():
                    while j < len(s) and s[j].isdigit():
                        j += 1
                    out.append(s[i:j])
                    i = j
                    continue
                k = i
                while k < len(s) and s[k] == "?":
                    k += 1
                run = k - i
                if run == 1:
                    out.append("?")
                else:
                    out.append(f"?{run}")
                i = k
                continue
            # 한글 또는 영어 소문자 (색 토큰)
            if re.match(r"[가-힣a-z]", ch):
                j = i + 1
                while j < len(s) and s[j].isdigit():
                    j += 1
                out.append(s[i:j])
                i = j
                continue
            raise ValueError(f"Invalid compact token at pos={i}: {s!r}")
        return out

    def expand_token(tok: str) -> List[str]:
        nonlocal next_q
        if tok == "?":
            t = f"?{next_q}"
            next_q += 1
            return [t]
        if tok.startswith("?"):
            n_str = tok[1:]
            if not n_str.isdigit():
                raise ValueError(f"Invalid unknown token: {tok!r}")
            n = int(n_str)
            if n <= 0:
                raise ValueError(f"Invalid unknown count in token: {tok!r}")
            out = []
            for _ in range(n):
                out.append(f"?{next_q}")
                next_q += 1
            return out
        mrep = re.match(r"^(.+?)(\d+)$", tok)
        if mrep:
            base = mrep.group(1).strip()
            k = int(mrep.group(2))
            if k <= 0:
                raise ValueError(f"Invalid repeat count in token: {tok!r}")
            return [normalize_color(base)] * k
        return [normalize_color(tok)]

    for idx1, line in enumerate(lines, 1):
        locked_here = False
        ws = line.strip().split()
        if ws and ws[0] == "고":
            locked_here = True
            ws = ws[1:]
        if ws and ws[-1] == "고":
            locked_here = True
            ws = ws[:-1]
        line2 = " ".join(ws).strip()
        if line2.startswith("고"):
            locked_here = True
            line2 = line2[1:].strip()

        rest_color: Optional[str] = None
        base_slash = line2

        if "/" in line2:
            base_slash, rest = line2.rsplit("/", 1)
            rest = rest.strip()
            if not rest:
                raise ValueError(f"Empty restriction color after '/' in line {idx1}: {line!r}")
            # f = freeze(잠금), 고와 동일
            if rest.lower() == "f":
                locked_here = True
            else:
                rest_color = rest

        base_slash = base_slash.strip()

        tube_cap = cap
        if base_slash.endswith("1x"):
            tube_cap = 1
            locked.add(idx1)
            base_slash = base_slash[:-2].strip()
        elif base_slash.endswith("1"):
            tube_cap = 1
            base_slash = base_slash[:-1].strip()

        if base_slash == "-":
            if rest_color is not None:
                raise ValueError(f"Inactive tube '-' cannot have restriction in line {idx1}: {line!r}")
            tubes.append(tuple())
            caps_1based.append(0)
            if locked_here:
                locked.add(idx1)
            continue

        m = re.match(r"^(.*)\(([^()]*)\)\s*$", base_slash)
        base = base_slash
        if m:
            if rest_color is not None:
                raise ValueError(f"Both '(...)' and '/...' restriction found in line {idx1}: {line!r}")
            base = m.group(1).strip()
            rest_color = m.group(2).strip()
            if not rest_color:
                raise ValueError(f"Empty restriction color in line {idx1}: {line!r}")

        if rest_color is not None and re.search(r"\d", rest_color):
            raise ValueError(
                f"Restriction color must not contain digits: {rest_color!r} (line {idx1})"
            )
        if rest_color is not None:
            rest_color = normalize_color(rest_color)

        parts = tokenize_base(base)
        norm: List[str] = []
        for p in parts:
            norm.extend(expand_token(p))

        if len(norm) > tube_cap:
            raise ValueError(f"Tube {idx1} exceeds CAP={tube_cap}: {norm}")

        if rest_color is not None:
            restricted[idx1] = rest_color
        if locked_here:
            locked.add(idx1)

        tubes.append(tuple(norm))
        caps_1based.append(tube_cap)

    return tuple(tubes), restricted, locked, caps_1based, mode


# ============================================================
# 게임 엔진
# ============================================================

def is_unknown(color: str) -> bool:
    """미지 블록인지 확인"""
    return color.startswith("?")


def top_is_unknown(tube: Tuple[str, ...]) -> bool:
    """병의 맨 위가 미지 블록인지"""
    return len(tube) > 0 and is_unknown(tube[0])


def can_pour(
    tube_caps: List[int],
    restricted_put: Dict[int, str],
    locked_out: Set[int],
    src: Tuple[str, ...],
    dst: Tuple[str, ...],
    src_i: int,
    dst_i: int,
) -> bool:
    """붓기 가능 여부 판정"""
    # 잠금 병에서는 꺼낼 수 없음
    if src_i in locked_out:
        return False
    # 출발 병이 비어있으면 안 됨
    if not src:
        return False
    # 미지 블록이 맨 위면 꺼낼 수 없음
    if top_is_unknown(src):
        return False
    # 도착 병이 꽉 차면 안 됨
    cap_dst = tube_caps[dst_i]
    if len(dst) >= cap_dst:
        return False
    # 도착 병의 맨 위가 미지 블록이면 부을 수 없음
    if top_is_unknown(dst):
        return False
    # 넣기 제한 확인
    c = src[0]
    if dst_i in restricted_put and restricted_put[dst_i] != c:
        return False
    # 같은 색 위 또는 빈 병에만
    return (not dst) or (dst[0] == c)


def pour(
    tube_caps: List[int],
    state: State,
    i0: int,
    j0: int,
) -> Optional[Tuple[State, int, str]]:
    """
    i0 -> j0 (0-based) 붓기.
    연속된 같은 색을 가능한 만큼 한 번에 부음.
    반환: (새 상태, 이동 개수, 색) 또는 None
    """
    src = list(state[i0])
    dst = list(state[j0])
    if not src:
        return None

    c = src[0]
    # 연속된 같은 색 개수
    k = 0
    while k < len(src) and src[k] == c:
        k += 1

    cap_dst = tube_caps[j0 + 1]
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


# ============================================================
# 점수 계산
# ============================================================

def chunk_count(tube: Tuple[str, ...]) -> int:
    """병의 청크 수. 빈 병은 0."""
    if not tube:
        return 0
    chunks = 1
    for i in range(1, len(tube)):
        if tube[i] != tube[i - 1]:
            chunks += 1
    return chunks


def count_colors(state: State) -> int:
    """상태에 존재하는 색의 종류 수 (미지 제외)"""
    colors = {c for tube in state for c in tube if not is_unknown(c)}
    return len(colors)


def count_empty(state: State) -> int:
    """빈 병 개수"""
    return sum(1 for t in state if not t)


def total_chunks(state: State) -> int:
    """전체 청크 수 합"""
    return sum(chunk_count(t) for t in state)


def score(state: State) -> Tuple[int, int]:
    """
    점수: (청크 수가 적을수록, 빈 병이 많을수록) 좋음
    → 비교용으로 (-청크수, 빈병수) 반환 (클수록 좋음)
    """
    return (-total_chunks(state), count_empty(state))


def is_optimal(state: State) -> bool:
    """최적 상태: 청크 수 = 색의 개수"""
    return total_chunks(state) == count_colors(state)


def is_solved(state: State, tube_caps: List[int]) -> bool:
    """승리 조건: 각 병이 비었거나 용량만큼 꽉 찬 단색"""
    for idx, t in enumerate(state, 1):
        if not t:
            continue
        if len(t) != tube_caps[idx]:
            return False
        if len(set(t)) != 1:
            return False
    return True


# ============================================================
# 미지 추정 (전체 모드)
# ============================================================

def get_unknowns(state: State) -> List[str]:
    """상태에서 미지 블록 토큰 목록 반환"""
    return [c for tube in state for c in tube if is_unknown(c)]


def get_known_colors(state: State) -> Set[str]:
    """상태에서 알려진 색 목록 반환"""
    return {c for tube in state for c in tube if not is_unknown(c)}


def color_counts_known(state: State) -> Dict[str, int]:
    """알려진 색의 개수 (미지 제외)"""
    counts: Dict[str, int] = {}
    for tube in state:
        for c in tube:
            if not is_unknown(c):
                counts[c] = counts.get(c, 0) + 1
    return counts


def infer_unknowns_full_mode(state: State) -> Optional[Dict[str, str]]:
    """
    전체 모드에서 미지 블록 추정.
    각 색은 정확히 CAP(4)개씩 존재해야 함.
    
    반환: {미지토큰: 색} 또는 None (추정 불가)
    """
    unknowns = get_unknowns(state)
    if not unknowns:
        return {}
    
    counts = color_counts_known(state)
    colors = get_known_colors(state)
    
    # 각 색에 대해 부족한 개수 계산
    needed: List[str] = []
    for c in colors:
        shortage = CAP - counts.get(c, 0)
        if shortage < 0:
            # 색이 4개 초과 -> 불가능
            return None
        needed.extend([c] * shortage)
    
    # 미지 개수와 필요 개수가 일치해야 함
    if len(needed) != len(unknowns):
        return None
    
    # 간단한 경우: 모든 미지에 순서대로 할당
    # (복잡한 조합 탐색은 나중에)
    from itertools import permutations
    
    # 조합이 너무 많으면 제한
    if len(unknowns) > 8:
        # 단순 할당 (첫 번째 조합만)
        assignment = dict(zip(unknowns, needed))
        return assignment
    
    # 모든 순열 시도 -> 해가 있는 것 반환 (여기선 첫 번째만)
    assignment = dict(zip(unknowns, needed))
    return assignment


def substitute_unknowns(state: State, assignment: Dict[str, str]) -> State:
    """미지 블록을 실제 색으로 치환"""
    new_tubes = []
    for tube in state:
        new_tube = tuple(assignment.get(c, c) for c in tube)
        new_tubes.append(new_tube)
    return tuple(new_tubes)


def generate_unknown_assignments(state: State) -> List[Dict[str, str]]:
    """
    전체 모드에서 가능한 미지 할당 조합 생성.
    조합이 너무 많으면 랜덤 샘플링.
    """
    import random
    from itertools import permutations
    
    unknowns = get_unknowns(state)
    if not unknowns:
        return [{}]
    
    counts = color_counts_known(state)
    colors = get_known_colors(state)
    
    # 각 색에 대해 부족한 개수
    needed: List[str] = []
    for c in sorted(colors):
        shortage = CAP - counts.get(c, 0)
        if shortage < 0:
            return []
        needed.extend([c] * shortage)
    
    if len(needed) != len(unknowns):
        return []
    
    # 중복 제거된 순열 생성
    seen: Set[Tuple[str, ...]] = set()
    all_assignments: List[Dict[str, str]] = []
    
    for perm in permutations(needed):
        if perm in seen:
            continue
        seen.add(perm)
        all_assignments.append(dict(zip(unknowns, perm)))
    
    # 100개 초과면 셔플 후 샘플링
    max_assignments = 100
    if len(all_assignments) > max_assignments:
        random.shuffle(all_assignments)
        return all_assignments[:max_assignments]
    
    return all_assignments


# ============================================================
# 탐색 (Beam Search)
# ============================================================

def solve(
    start: State,
    tube_caps: List[int],
    restricted_put: Dict[int, str],
    locked_out: Set[int],
    beam_width: int = 8000,
    max_steps: int = 400,
    time_limit_sec: float = 30.0,
) -> dict:
    """
    최대 점수 상태를 찾는 beam search.
    """
    t0 = time.time()
    N = len(start)

    best_state = start
    best_score = score(start)
    best_path: List[Tuple[int, int, int, str]] = []

    def neg_sc(sc: Tuple[int, int]) -> Tuple[int, int]:
        return (-sc[0], -sc[1])

    # 우선순위 큐: (neg_score, steps, state)
    pq: List[Tuple[Tuple[int, int], int, State]] = []
    heapq.heappush(pq, (neg_sc(best_score), 0, start))

    parent: Dict[State, Optional[State]] = {start: None}
    how: Dict[State, Optional[Tuple[int, int, int, str]]] = {start: None}
    seen_best: Dict[State, Tuple[Tuple[int, int], int]] = {start: (best_score, 0)}

    visited = 0

    while pq and (time.time() - t0) < time_limit_sec:
        _, steps, cur = heapq.heappop(pq)
        visited += 1

        cur_sc = score(cur)
        if cur_sc > best_score:
            best_score = cur_sc
            best_state = cur
            # 경로 복원
            path: List[Tuple[int, int, int, str]] = []
            x = cur
            while how.get(x) is not None:
                path.append(how[x])  # type: ignore
                x = parent[x]  # type: ignore
            path.reverse()
            best_path = path

        # 최적 상태 도달 시 조기 종료
        if is_optimal(cur):
            break

        if steps >= max_steps:
            continue

        # 다음 상태 생성
        nxt_candidates: List[Tuple[Tuple[int, int], int, int, int, str, State]] = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if not can_pour(tube_caps, restricted_put, locked_out, cur[i], cur[j], i + 1, j + 1):
                    continue
                res = pour(tube_caps, cur, i, j)
                if res is None:
                    continue
                nxt, moved, color = res
                sc = score(nxt)
                nxt_candidates.append((sc, i + 1, j + 1, moved, color, nxt))

        # beam width 제한
        nxt_candidates.sort(key=lambda x: x[0], reverse=True)
        if len(nxt_candidates) > beam_width:
            nxt_candidates = nxt_candidates[:beam_width]

        for sc, a, b, m, c, nxt in nxt_candidates:
            prev = seen_best.get(nxt)
            if (prev is None) or (sc > prev[0]) or (sc == prev[0] and steps + 1 < prev[1]):
                seen_best[nxt] = (sc, steps + 1)
                parent[nxt] = cur
                how[nxt] = (a, b, m, c)
                heapq.heappush(pq, (neg_sc(sc), steps + 1, nxt))

    elapsed = time.time() - t0
    return {
        "visited": visited,
        "elapsed_sec": elapsed,
        "best_state": best_state,
        "best_path": best_path,
        "total_chunks": total_chunks(best_state),
        "empty_tubes": count_empty(best_state),
        "color_count": count_colors(best_state),
        "is_optimal": is_optimal(best_state),
        "is_solved": is_solved(best_state, tube_caps),
    }


# ============================================================
# 출력
# ============================================================

def print_state(
    state: State,
    tube_caps: List[int],
    restricted_put: Dict[int, str],
    locked_out: Set[int],
) -> None:
    for idx, tube in enumerate(state, 1):
        tags = []
        if idx in restricted_put:
            tags.append(f"넣기 제한: {restricted_put[idx]}만")
        if idx in locked_out:
            tags.append("고")
        if tube_caps[idx] != CAP:
            tags.append(f"{tube_caps[idx]}칸")
        tag = f" ({', '.join(tags)})" if tags else ""
        
        # 청크 수 표시
        chunks = chunk_count(tube)
        print(f"{idx:2d}: {list(tube)} [청크:{chunks}]{tag}")


def print_path(path: List[Tuple[int, int, int, str]]) -> None:
    for k, (a, b, m, c) in enumerate(path, 1):
        print(f"{k:03d}. {a} -> {b} : {c} x{m}")


# ============================================================
# Ray 병렬 탐색
# ============================================================

def _solve_one_assignment(
    assignment: Dict[str, str],
    start: State,
    tube_caps: List[int],
    restricted_put: Dict[int, str],
    locked_out: Set[int],
    beam_width: int,
    max_steps: int,
    time_limit_sec: float,
) -> Tuple[Dict[str, str], dict]:
    """단일 미지 할당에 대해 solve 실행 (Ray용)"""
    subst_state = substitute_unknowns(start, assignment)
    result = solve(
        subst_state,
        tube_caps,
        restricted_put,
        locked_out,
        beam_width=beam_width,
        max_steps=max_steps,
        time_limit_sec=time_limit_sec,
    )
    return assignment, result


def solve_assignments_ray(
    assignments: List[Dict[str, str]],
    start: State,
    tube_caps: List[int],
    restricted_put: Dict[int, str],
    locked_out: Set[int],
    beam_width: int = 10000,
    max_steps: int = 500,
    time_limit_sec: float = 30.0,
    max_procs: int = 8,
) -> List[Tuple[Dict[str, str], dict]]:
    """
    Ray로 미지 할당 조합들을 병렬 탐색.
    모든 (assignment, result) 리스트 반환.
    """
    try:
        import ray
    except ImportError:
        # Ray 없으면 순차 실행
        print("(Ray 미설치 - 순차 실행)")
        results = []
        for assignment in assignments:
            subst_state = substitute_unknowns(start, assignment)
            result = solve(
                subst_state,
                tube_caps,
                restricted_put,
                locked_out,
                beam_width=beam_width,
                max_steps=max_steps,
                time_limit_sec=time_limit_sec / len(assignments),
            )
            results.append((assignment, result))
        return results

    # Ray 초기화 (로깅 억제)
    import logging
    logging.getLogger("ray").setLevel(logging.ERROR)
    ray.init(
        num_cpus=max_procs,
        ignore_reinit_error=True,
        include_dashboard=False,
        configure_logging=True,
        logging_level=logging.ERROR,
    )
    
    try:
        # 원격 함수 정의
        @ray.remote
        def remote_solve(
            assignment: Dict[str, str],
            start: State,
            tube_caps: List[int],
            restricted_put: Dict[int, str],
            locked_out: Set[int],
            beam_width: int,
            max_steps: int,
            time_limit_sec: float,
        ):
            return _solve_one_assignment(
                assignment, start, tube_caps, restricted_put, locked_out,
                beam_width, max_steps, time_limit_sec
            )
        
        # 태스크 제출
        refs = [
            remote_solve.remote(
                assignment, start, tube_caps, restricted_put, locked_out,
                beam_width, max_steps, time_limit_sec
            )
            for assignment in assignments
        ]
        
        # 결과 수집
        return ray.get(refs)
    
    finally:
        ray.shutdown()


def analyze_first_moves(
    all_results: List[Tuple[Dict[str, str], dict]],
) -> List[Tuple[Tuple[int, int, str], int, float]]:
    """
    모든 해법에서 첫 번째 이동을 수집하고 확률 계산.
    
    반환: [(이동(src, dst, color), 횟수, 확률), ...] 내림차순
    """
    from collections import Counter
    
    first_moves: List[Tuple[int, int, str]] = []
    total_with_path = 0
    
    for assignment, result in all_results:
        path = result.get("best_path", [])
        if path:
            # path[0] = (src, dst, count, color)
            src, dst, _, color = path[0]
            first_moves.append((src, dst, color))
            total_with_path += 1
    
    if total_with_path == 0:
        return []
    
    counter = Counter(first_moves)
    ranked = [
        (move, count, count / total_with_path)
        for move, count in counter.most_common()
    ]
    return ranked


# ============================================================
# 메인
# ============================================================

INITIAL_STATE_RAW = """

w2ls
e2v
ryo
x

vewl
rsoy
-
wsor
osyl

""".strip()


if __name__ == "__main__":
    try:
        # 파싱
        start, restricted_put, locked_out, tube_caps, mode = parse_initial_state(INITIAL_STATE_RAW)
        N = len(start)

        print("=== 초기 상태 ===")
        print(f"모드: {mode}")
        print_state(start, tube_caps, restricted_put, locked_out)
        print()
        print(f"색 개수: {count_colors(start)}")
        print(f"초기 청크 수: {total_chunks(start)}")
        print(f"초기 빈 병: {count_empty(start)}")
        print(f"초기 점수: {score(start)}")
        
        unknowns = get_unknowns(start)
        if unknowns:
            print(f"미지 블록: {unknowns}")
        print()

        # 설정
        beam_width = int(os.getenv("BEAM_WIDTH", "10000"))
        max_steps = int(os.getenv("MAX_STEPS", "500"))
        time_limit = float(os.getenv("TIME_LIMIT", "30.0"))

        # 설정: Ray 프로세스 수
        max_procs = int(os.getenv("MAX_PROCS", "8"))

        # 전체 모드이고 미지가 있으면 분석
        if mode == "full" and unknowns:
            print("=== 전체 모드: 미지 블록 분석 ===")
            counts = color_counts_known(start)
            print(f"현재 색 개수: {counts}")
            
            assignments = generate_unknown_assignments(start)
            if not assignments:
                print("미지 블록 추정 불가 (색 개수 조건 불일치)")
            else:
                print(f"가능한 조합 수: {len(assignments)}")
                print(f"병렬 탐색 (Ray, max_procs={max_procs})")
                print()
                
                # Ray 병렬 탐색 - 모든 결과 수집
                all_results = solve_assignments_ray(
                    assignments,
                    start,
                    tube_caps,
                    restricted_put,
                    locked_out,
                    beam_width=beam_width,
                    max_steps=max_steps,
                    time_limit_sec=time_limit,
                    max_procs=max_procs,
                )
                
                # 해법 통계
                solved_count = sum(1 for _, r in all_results if r.get("is_solved"))
                optimal_count = sum(1 for _, r in all_results if r.get("is_optimal"))
                has_path_count = sum(1 for _, r in all_results if r.get("best_path"))
                
                print(f"=== 조합별 해법 통계 ===")
                print(f"전체 조합: {len(all_results)}")
                print(f"승리 가능: {solved_count} ({100*solved_count/len(all_results):.1f}%)")
                print(f"최적 도달: {optimal_count} ({100*optimal_count/len(all_results):.1f}%)")
                print(f"이동 가능: {has_path_count} ({100*has_path_count/len(all_results):.1f}%)")
                print()
                
                # 100% 수 연속 분석
                certain_moves: List[Tuple[int, int, int, str]] = []
                remaining_results = all_results
                step = 0
                
                while True:
                    step += 1
                    first_move_stats = analyze_first_moves(remaining_results)
                    
                    if not first_move_stats:
                        break
                    
                    # 100%가 아니면 여기서 출력하고 종료
                    top_move, top_count, top_prob = first_move_stats[0]
                    
                    if top_prob < 1.0:
                        print(f"=== {step}번째 수 후보 (확률 내림차순) ===")
                        for (src, dst, color), count, prob in first_move_stats:
                            bar = "█" * int(prob * 20)
                            print(f"  {src} -> {dst} ({color}): {count}회 ({100*prob:.1f}%) {bar}")
                        break
                    
                    # 100%면 확정 수로 추가
                    src, dst, color = top_move
                    # 실제 이동 정보 (count 포함) 찾기
                    for assignment, result in remaining_results:
                        path = result.get("best_path", [])
                        if path:
                            certain_moves.append(path[0])
                            break
                    
                    # 각 해법에서 첫 수 제거하여 다음 수 분석 준비
                    new_remaining = []
                    for assignment, result in remaining_results:
                        path = result.get("best_path", [])
                        if path and len(path) > 1:
                            new_result = dict(result)
                            new_result["best_path"] = path[1:]
                            new_remaining.append((assignment, new_result))
                    
                    if not new_remaining:
                        break
                    
                    remaining_results = new_remaining
                
                # 확정 수 출력
                if certain_moves:
                    print(f"=== 확정 수 (100% 확률) ===")
                    for i, (src, dst, cnt, color) in enumerate(certain_moves, 1):
                        print(f"  {i:03d}. {src} -> {dst} : {color} x{cnt}")
                    print()
                
                raise SystemExit(0)

        print(f"=== 탐색 시작 (beam={beam_width}, max_steps={max_steps}, time={time_limit}s) ===")
        result = solve(
            start,
            tube_caps,
            restricted_put,
            locked_out,
            beam_width=beam_width,
            max_steps=max_steps,
            time_limit_sec=time_limit,
        )

        print()
        print(f"=== 탐색 결과 ({result['elapsed_sec']:.2f}s, visited={result['visited']}) ===")
        print(f"최종 청크 수: {result['total_chunks']} (목표: {result['color_count']})")
        print(f"빈 병 개수: {result['empty_tubes']}")
        print(f"최적 상태: {'예' if result['is_optimal'] else '아니오'}")
        print(f"승리 상태: {'예' if result['is_solved'] else '아니오'}")
        print()

        path = result["best_path"]
        print(f"=== 이동 순서 ({len(path)}수) ===")
        if path:
            print_path(path)
        else:
            print("(이동 없음)")

        print()
        print("=== 최종 상태 ===")
        print_state(result["best_state"], tube_caps, restricted_put, locked_out)

    except BrokenPipeError:
        pass
