from __future__ import annotations
from collections import deque, defaultdict
from typing import Tuple, List, Dict, Set, Optional
import heapq
import time
import os
import re
import _engine as engine
import _scoring as scoring

CAP = 4
N = 10

# 색 총개수 가정(미지 추정/불가능 판정에 영향)
# - "full": 전체문제. 각 색은 정확히 CAP개(=4개)라고 가정 (엄격 완료 판정에서 count==CAP만 허용)
# - "partial": 부분문제. 각 색은 CAP개 이하일 수 있음 (엄격 완료 판정에서 count<=CAP만 허용)
COLOR_TOTAL_MODE = os.getenv("COLOR_TOTAL_MODE", "full").strip().lower()
if COLOR_TOTAL_MODE not in ("full", "partial"):
    COLOR_TOTAL_MODE = "full"

INITIAL_STATE_RAW = """
(부분)
회???
회연갈
노회갈

파핑파
적2주하
회적녹
녹2연하
하노파
노

""".strip()

locked_out: Set[int] = set()
no_put: Set[int] = set()
# 1-based index용 cap 배열 (index 0은 더미). 기본은 4칸.
tube_caps: List[int] = [0]


def parse_initial_state(raw: str, cap: int = CAP) -> tuple[State, Dict[int, str], Set[int], List[int], Set[int], str]:
    """
    초기상태를 텍스트로 입력하기 위한 파서.

    형식:
      - 각 줄이 병 1개(위->아래)
      - 쉼표(,)로 색을 나열
      - 병 끝에 괄호가 있으면 그 병은 '(색)'만 다시 넣기 가능
        예) 노랑,하늘,자색,자색(자색)
      - 병에 '고' 표시가 있으면, 그 병은 "넣기는 가능하지만 꺼내기는 불가"
        예) 고주보연회   /   주보연회 고
      - 라인 끝의 접미사:
        - '1'  : 1칸짜리 병
        - '1x' : 1칸짜리 병 + 꺼내기 불가(=src로 사용 불가, 넣기는 가능)
      - '?'는 미지색 토큰이며, 등장 순서대로 ?1, ?2, ... 로 자동 치환
    """
    if not raw.strip():
        raise ValueError("initial state text is empty")

    lines: List[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        lines.append(s)

    # 모드 헤더: 맨 첫줄이 "(부분)"이면 partial, 아니면 full.
    # (헤더가 없을 때만 환경변수 COLOR_TOTAL_MODE를 fallback으로 사용)
    mode = os.getenv("COLOR_TOTAL_MODE", "full").strip().lower()
    if lines and lines[0] == "(부분)":
        mode = "partial"
        lines = lines[1:]
    else:
        mode = "full" if mode not in ("full", "partial") else mode
        if mode not in ("full", "partial"):
            mode = "full"

    restricted: Dict[int, str] = {}
    tubes: List[Tuple[str, ...]] = []
    locked: Set[int] = set()
    no_put_local: Set[int] = set()
    caps_1based: List[int] = [0]
    next_q = 1
    unknowns: List[str] = []

    def normalize_color(tok: str) -> str:
        """색 별칭 정규화. 예: '레'/'적'은 '빨', '청'은 '파'로 취급."""
        if tok in ("레", "적"):
            return "빨"
        if tok == "청":
            return "파"
        return tok

    def tokenize_base(base: str) -> List[str]:
        """
        병의 내용(base)을 토큰 리스트로 분해.

        지원 형식:
          - CSV:  '주황,보라,연두,회색'
          - Compact(권장): '주보연회'  (색은 1글자씩)
            - 반복: '녹2' => '녹','녹'
            - 미지: '?', '?3', '??'(= '?2'), '???'(= '?3')
          - 빈 병: '빈'  => []
        """
        s = "".join(base.split())  # 모든 공백 제거
        if not s:
            return []
        if s == "빈":
            return []
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) == 1 and parts[0] == "빈":
                return []
            return parts

        # compact: 한 글자 색(가-힣) 또는 '?' + (선택) 숫자
        out: List[str] = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "?":
                # 1) '?<digits>' 형태면 그걸 우선
                j = i + 1
                if j < len(s) and s[j].isdigit():
                    while j < len(s) and s[j].isdigit():
                        j += 1
                    out.append(s[i:j])  # '?3'
                    i = j
                    continue
                # 2) 연속된 '??'는 '?N'과 동일 취급
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
            if re.match(r"[가-힣]", ch):
                j = i + 1
                while j < len(s) and s[j].isdigit():
                    j += 1
                out.append(s[i:j])  # '주' 또는 '녹2'
                i = j
                continue
            raise ValueError(f"Invalid compact token at pos={i}: {s!r}")
        return out


    def expand_token(tok: str) -> List[str]:
        """
        토큰 확장 규칙:
          - "?"  => 새로운 ?n 1개 부여
          - "?3" => 새로운 ?n 3개 부여(서로 다른 미지토큰 3개)
          - "노랑2" => ["노랑", "노랑"]
          - "녹2" => ["녹", "녹"]
          - 그 외 => ["토큰"]
        """
        nonlocal next_q
        if tok == "?":
            t = f"?{next_q}"
            next_q += 1
            return [t]
        if tok.startswith("?"):
            # "?3" => 미지토큰 3개 생성
            n_str = tok[1:]
            if not n_str.isdigit():
                raise ValueError(f"Invalid unknown token syntax: {tok!r} (use '?' or '?<number>')")
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
        # '고' 표기(잠금 병): 넣기는 가능하지만 꺼내기 불가(=src로 사용 불가)
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
            # compact에서 "고주보연회" 같은 표기 지원
            locked_here = True
            line2 = line2[1:].strip()

        # 제한 표기 지원:
        #  - 괄호:   "노랑,하늘,자색,자색(자색)"
        #  - 슬래시: "하녹2갈/갈"  (오른쪽은 '그 색만 다시 넣기 가능')
        rest_color: Optional[str] = None

        base_slash = line2
        if "/" in line2:
            base_slash, rest = line2.rsplit("/", 1)
            rest = rest.strip()
            if not rest:
                raise ValueError(f"Empty restriction color after '/' in line {idx1}: {line!r}")
            rest_color = rest

        base_slash = base_slash.strip()

        # 한칸짜리 병 표기: 라인 끝의 '1' 또는 '1x'
        #  - 연1  : 1칸 병
        #  - 빈1  : 1칸 빈 병
        #  - 연1x : 1칸 병 + 꺼내기 불가(=locked)
        tube_cap = cap
        if base_slash.endswith("1x"):
            tube_cap = 1
            locked.add(idx1)
            base_slash = base_slash[:-2].strip()
        elif base_slash.endswith("1"):
            tube_cap = 1
            base_slash = base_slash[:-1].strip()

        # '-'만 있는 줄: 병 번호만 존재하는 비활성 병(0칸).
        # (색이 들어있을 수도 없고, 넣을 수도 없음. 번호 맞추기 용도)
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
            raise ValueError(f"Restriction color must not contain digits: {rest_color!r} (line {idx1})")
        if rest_color is not None:
            rest_color = normalize_color(rest_color)

        parts = tokenize_base(base)
        norm: List[str] = []
        for p in parts:
            expanded = expand_token(p)
            norm.extend(expanded)
            for x in expanded:
                if x.startswith("?"):
                    unknowns.append(x)

        if len(norm) > tube_cap:
            raise ValueError(f"Tube {idx1} exceeds CAP={tube_cap}: {norm}")

        if rest_color is not None:
            restricted[idx1] = rest_color
        if locked_here:
            locked.add(idx1)

        tubes.append(tuple(norm))
        caps_1based.append(tube_cap)

    # 미지색 토큰은 서로 달라야 함
    dups = {u for u in unknowns if unknowns.count(u) > 1}
    if dups:
        raise ValueError(f"Unknown tokens must be distinct, duplicates found: {sorted(dups)}")

    return tuple(tubes), restricted, locked, caps_1based, no_put_local, mode


# 초기 상태/제한은 위의 텍스트 블록에서 파싱해 구성합니다.
# (단, INITIAL_STATE_RAW가 비어있으면 import가 깨지지 않도록 빈 상태로 둡니다.
#  다른 스크립트에서 parse_initial_state 결과로 전역을 덮어쓸 수 있습니다.)
if INITIAL_STATE_RAW.strip():
    start, restricted_put, locked_out, tube_caps, no_put, COLOR_TOTAL_MODE = parse_initial_state(INITIAL_STATE_RAW, cap=CAP)
    N = len(start)
else:
    start = tuple()
    restricted_put = {}
    locked_out = set()
    no_put = set()
    tube_caps = [0]
    N = 0

State = Tuple[Tuple[str, ...], ...]


def can_put(dst_idx_1based: int, color: str) -> bool:
    """목적지 병에 해당 색을 '붓는 것'이 허용되는지."""
    return engine.can_put(restricted_put, dst_idx_1based, color)


def can_pour(src: Tuple[str, ...], dst: Tuple[str, ...], src_i: int, dst_i: int) -> bool:
    """같은색 위 또는 빈병, + 목적지 병의 '넣기 제한' 준수."""
    return engine.can_pour(tube_caps, restricted_put, locked_out, no_put, src, dst, src_i, dst_i)


def pour(state: State, i0: int, j0: int) -> Optional[tuple[State, int, str]]:
    """
    i0->j0 (0-based). src top의 연속 같은색을 가능한 만큼 한 번에 붓기.
    반환: (새 상태, 이동칸수, 색) 또는 None
    """
    return engine.pour(tube_caps, state, i0, j0)


def reachable_with_color(
    target_color: str,
    goal_kind: str = "have_4_stack",
    max_nodes: int = 100_000_000,
) -> dict:
    """
    특정 색 target_color에 대해 '가능성'을 탐색으로 확인.

    goal_kind:
      - "have_4_stack": target_color로 된 4칸 단색 병이 '어느 상태에서든' 존재 가능한가?
      - "top_in_tube": target_color가 어떤 병의 맨 위에 오게 할 수 있는가?
      - "move_possible": target_color를 한 번이라도 실제로 '붓기(이동)' 할 수 있는가?
        (예: ? 토큰이 target_color이면 그 색이 실제 이동되는지)

    max_nodes: BFS 방문 상태 상한(너무 커지는 것 방지)

    반환 dict:
      {
        "possible": bool,
        "visited": int,
        "witness_state": State|None,
        "witness_path": List[(from,to,amount,color)]|None,
        "note": str
      }
    """
    def is_goal(state: State) -> bool:
        if goal_kind == "have_4_stack":
            for t in state:
                if len(t) == CAP and len(set(t)) == 1 and t[0] == target_color:
                    return True
            return False
        elif goal_kind == "top_in_tube":
            return any(t and t[0] == target_color for t in state)
        elif goal_kind == "move_possible":
            # 목표를 "target_color가 이동된 적이 있는지"로 잡기 위해,
            # BFS에서는 이동정보를 보고 판정할 것이라 여기서는 False
            return False
        else:
            raise ValueError(f"Unknown goal_kind={goal_kind}")

    q = deque([start])
    parent: Dict[State, Optional[State]] = {start: None}
    how: Dict[State, Optional[tuple[int,int,int,str]]] = {start: None}

    visited = 0
    moved_target_once = False
    witness_state: Optional[State] = None

    while q and visited < max_nodes:
        cur = q.popleft()
        visited += 1

        # goal_kind가 move_possible인 경우는 "이동"을 보면서 체크해야 함
        if goal_kind != "move_possible" and is_goal(cur):
            witness_state = cur
            break

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if not can_pour(cur[i], cur[j], i+1, j+1):
                    continue
                res = pour(cur, i, j)
                if res is None:
                    continue
                nxt, moved, color = res

                # 목표: target_color가 실제로 이동되는가?
                if goal_kind == "move_possible" and color == target_color:
                    moved_target_once = True
                    witness_state = nxt

                if nxt not in parent:
                    parent[nxt] = cur
                    how[nxt] = (i+1, j+1, moved, color)
                    q.append(nxt)

                if witness_state is not None and (goal_kind == "move_possible"):
                    # 이동이 발생한 즉시 멈추기(원하면 더 깊게 탐색하도록 수정 가능)
                    q.clear()
                    break
            if witness_state is not None and (goal_kind == "move_possible"):
                break

    if witness_state is None:
        return {
            "possible": False,
            "visited": visited,
            "witness_state": None,
            "witness_path": None,
            "note": f"max_nodes={max_nodes} 내에서는 목표 달성 상태를 찾지 못했습니다. "
                    f"(불가능일 수도 있고, 탐색 한계일 수도 있습니다.)"
        }

    # 경로 복원
    path: List[tuple[int,int,int,str]] = []
    x = witness_state
    while how[x] is not None:
        path.append(how[x])
        x = parent[x]  # type: ignore
    path.reverse()

    return {
        "possible": True,
        "visited": visited,
        "witness_state": witness_state,
        "witness_path": path,
        "note": "목표 달성 상태를 찾았습니다."
    }


def reachable_with_color_multi(
    target_color: str,
    goal_kind: str = "have_4_stack",
    max_nodes: int = 5_000_000,
    max_witnesses: int = 10,
    stop_on_strict_solve: bool = False,
) -> dict:
    """
    reachable_with_color의 확장 버전: 목표를 달성하는 witness를 '여러 개' 수집.

    goal_kind:
      - "have_4_stack": target_color 4칸 단색 병이 존재하는 상태를 witness로 수집
      - "top_in_tube": target_color가 맨 위인 병이 존재하는 상태를 witness로 수집
      - "move_possible": target_color가 실제로 '이동'된 순간(엣지)을 witness로 수집

    반환 dict:
      {
        "possible": bool,
        "visited": int,
        "witnesses": List[{ "state": State, "path": List[(from,to,amount,color)] }],
        "note": str
      }
    """
    if max_witnesses <= 0:
        raise ValueError("max_witnesses must be >= 1")

    def is_goal_state(state: State) -> bool:
        if goal_kind == "have_4_stack":
            for t in state:
                if len(t) == CAP and len(set(t)) == 1 and t[0] == target_color:
                    return True
            return False
        elif goal_kind == "top_in_tube":
            return any(t and t[0] == target_color for t in state)
        elif goal_kind == "move_possible":
            return False  # 엣지를 보고 판단
        else:
            raise ValueError(f"Unknown goal_kind={goal_kind}")

    q = deque([start])
    parent: Dict[State, Optional[State]] = {start: None}
    how: Dict[State, Optional[tuple[int, int, int, str]]] = {start: None}

    visited = 0
    witness_states_seen: Set[State] = set()
    witnesses: List[dict] = []

    def reconstruct(end_state: State) -> List[tuple[int, int, int, str]]:
        path: List[tuple[int, int, int, str]] = []
        x = end_state
        while how[x] is not None:
            path.append(how[x])  # type: ignore[arg-type]
            x = parent[x]  # type: ignore[assignment]
        path.reverse()
        return path

    def strict_solved_info(state: State) -> Optional[dict]:
        if not stop_on_strict_solve:
            return None
        if is_solved_strict(state):
            return {
                "strict_solved": True,
                "strict_state": state,
                "strict_path": reconstruct(state),
                "strict_note": "엄격 완료 상태를 탐색 중 발견하여 중단합니다.",
            }
        return None

    # 시작 상태도 goal이면 포함(단, move_possible 제외)
    if goal_kind != "move_possible" and is_goal_state(start):
        witness_states_seen.add(start)
        witnesses.append({"state": start, "path": []})
        if len(witnesses) >= max_witnesses:
            return {
                "possible": True,
                "visited": 1,
                "witnesses": witnesses,
                "note": f"witness {len(witnesses)}개를 수집했습니다(시작 상태 포함).",
            }

    while q and visited < max_nodes and len(witnesses) < max_witnesses:
        cur = q.popleft()
        visited += 1

        si = strict_solved_info(cur)
        if si is not None:
            return {
                "possible": len(witnesses) > 0,
                "visited": visited,
                "witnesses": witnesses,
                "note": f"witness {len(witnesses)}개를 수집했습니다.",
                **si,
            }

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if not can_pour(cur[i], cur[j], i + 1, j + 1):
                    continue
                res = pour(cur, i, j)
                if res is None:
                    continue
                nxt, moved, color = res

                # move_possible은 "target_color가 실제로 이동된 엣지"를 witness로
                if goal_kind == "move_possible" and color == target_color:
                    if nxt not in witness_states_seen:
                        witness_states_seen.add(nxt)
                        witnesses.append({"state": nxt, "path": reconstruct(cur) + [(i + 1, j + 1, moved, color)]})
                        if len(witnesses) >= max_witnesses:
                            break

                if nxt not in parent:
                    parent[nxt] = cur
                    how[nxt] = (i + 1, j + 1, moved, color)
                    q.append(nxt)

                    si2 = strict_solved_info(nxt)
                    if si2 is not None:
                        return {
                            "possible": len(witnesses) > 0,
                            "visited": visited,
                            "witnesses": witnesses,
                            "note": f"witness {len(witnesses)}개를 수집했습니다.",
                            **si2,
                        }

                    if goal_kind != "move_possible" and is_goal_state(nxt):
                        if nxt not in witness_states_seen:
                            witness_states_seen.add(nxt)
                            witnesses.append({"state": nxt, "path": reconstruct(nxt)})
                            if len(witnesses) >= max_witnesses:
                                break
            if len(witnesses) >= max_witnesses:
                break

    possible = len(witnesses) > 0
    note = (
        f"witness {len(witnesses)}개를 수집했습니다."
        if possible
        else f"max_nodes={max_nodes} 내에서는 witness를 찾지 못했습니다."
    )
    return {
        "possible": possible,
        "visited": visited,
        "witnesses": witnesses,
        "note": note,
    }


def reachable_multi_colors_have_full_stack(max_nodes: int = 5_000_000, stop_on_strict_solve: bool = False) -> dict:
    """
    목표: 한 상태에서 '서로 다른 k개 색'에 대해 '꽉 찬 단색 병(=full stack)'이 동시에 존재하는지,
    k=2,3,4,... 에 대한 witness를 BFS 1번으로 수집.

    - full stack: 해당 병 길이 == tube_caps[idx] 이고 단색(len(set)==1)

    반환 dict:
      {
        "visited": int,
        "max_k": int,
        "witness_by_k": { k: { "colors": tuple[str,...], "state": State, "path": [...] } },
        "note": str
      }
    """
    q = deque([start])
    parent: Dict[State, Optional[State]] = {start: None}
    how: Dict[State, Optional[tuple[int, int, int, str]]] = {start: None}

    def full_stack_colors(state: State) -> Set[str]:
        out: Set[str] = set()
        for idx, t in enumerate(state, 1):
            if t and len(t) == tube_caps[idx] and len(set(t)) == 1:
                out.add(t[0])
        return out

    def reconstruct(end_state: State) -> List[tuple[int, int, int, str]]:
        path: List[tuple[int, int, int, str]] = []
        x = end_state
        while how[x] is not None:
            path.append(how[x])  # type: ignore[arg-type]
            x = parent[x]  # type: ignore[assignment]
        path.reverse()
        return path

    witness_by_k: Dict[int, dict] = {}
    visited = 0
    max_k = 0

    # 시작 상태 체크
    cols0 = sorted(full_stack_colors(start))
    if cols0:
        max_k = len(cols0)
        for k in range(2, max_k + 1):
            witness_by_k[k] = {"colors": tuple(cols0[:k]), "state": start, "path": []}
    if stop_on_strict_solve and is_solved_strict(start):
        return {
            "visited": 1,
            "max_k": max_k,
            "witness_by_k": witness_by_k,
            "strict_solved": True,
            "strict_state": start,
            "strict_path": [],
            "note": "시작 상태에서 이미 엄격 완료입니다.",
        }

    while q and visited < max_nodes:
        cur = q.popleft()
        visited += 1

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if not can_pour(cur[i], cur[j], i + 1, j + 1):
                    continue
                res = pour(cur, i, j)
                if res is None:
                    continue
                nxt, moved, color = res
                if nxt in parent:
                    continue
                parent[nxt] = cur
                how[nxt] = (i + 1, j + 1, moved, color)

                if stop_on_strict_solve and is_solved_strict(nxt):
                    return {
                        "visited": visited,
                        "max_k": max_k,
                        "witness_by_k": witness_by_k,
                        "strict_solved": True,
                        "strict_state": nxt,
                        "strict_path": reconstruct(nxt),
                        "note": "엄격 완료 상태를 탐색 중 발견하여 중단합니다.",
                    }

                cols = sorted(full_stack_colors(nxt))
                k_now = len(cols)
                if k_now > max_k:
                    max_k = k_now
                # BFS라서 각 k의 첫 발견이 최단 경로 witness
                for k in range(2, k_now + 1):
                    if k not in witness_by_k:
                        witness_by_k[k] = {"colors": tuple(cols[:k]), "state": nxt, "path": reconstruct(nxt)}

                q.append(nxt)

    exhausted = not q
    note = (
        f"탐색 가능한 모든 상태(visited={visited}) 내에서 max_k={max_k}까지 확인했습니다."
        if exhausted
        else f"max_nodes={max_nodes} 내에서 max_k={max_k}까지 확인했습니다. (visited={visited})"
    )
    return {"visited": visited, "max_k": max_k, "witness_by_k": witness_by_k, "note": note}


def print_state(state: State) -> None:
    engine.print_state(state, restricted_put, locked_out, no_put=no_put, tube_caps_1based=tube_caps)


def colors_in_state(state: State) -> List[str]:
    return sorted({c for tube in state for c in tube})


def color_counts(state: State) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tube in state:
        for c in tube:
            counts[c] = counts.get(c, 0) + 1
    return counts


def is_solved_strict(state: State) -> bool:
    """엄격 완료: 각 병이 비었거나(0칸) 4칸 단색."""
    for idx, t in enumerate(state, 1):
        if not t:
            continue
        if len(t) != tube_caps[idx]:
            return False
        if len(set(t)) != 1:
            return False
    return True


def is_solved_relaxed(state: State) -> bool:
    """완화 완료: 각 병이 비었거나 단색(길이 무관)."""
    for t in state:
        if not t:
            continue
        if len(set(t)) != 1:
            return False
    return True


def score_relaxed(state: State) -> tuple[int, int, int]:
    """
    beam search용 점수:
      1) 단색(또는 빈 병) 병 개수(크면 좋음)
      2) 완성된 4칸 단색 병 개수(크면 좋음)
      3) 응집도(각 병에서 가장 많은 색의 개수 합)(크면 좋음)
    """
    return scoring.score_relaxed(state, tube_caps_1based=tube_caps)


def _tube_chunk_count(tube: Tuple[str, ...]) -> int:
    """연속 덩어리(청크) 개수. 빈 병은 0. 예: 파파노노=2, 파노파노=4"""
    if not tube:
        return 0
    chunks = 1
    for i in range(1, len(tube)):
        if tube[i] != tube[i - 1]:
            chunks += 1
    return chunks


def recommendation_metrics(state: State) -> tuple[int, int, int, int, int]:
    """
    최종추천 평가 기준(사용자 정의):
      1) 전체 청크 수 합이 적을수록 우세 (minimize)
      2) (1 동률) '가려진 미지(?)'가 적을수록 우세 (minimize)
         - 미지는 그 위의 색이 사라져서 맨 위로 올라오면 색을 알 수 있다고 가정
         - 따라서 "맨 위가 아닌 위치에 있는 ?"만 카운트
      3) (1,2 동률) 빈 병이 많을수록 우세 (maximize)
      4) (1~3 동률) 서로 분리된 색이 적을수록 우세 (minimize)
         - 여기서 "분리된 색" = 같은 색이 2개 이상의 병에 걸쳐 존재하는 색의 개수
      5) (1~4 동률) 각 병의 바닥색(맨 아래)을 나열했을 때
         중복이 없을수록 우세 (maximize distinct count)

    반환: (chunk_sum, hidden_unknowns, empty_tubes, separated_colors, distinct_bottoms)
    """
    chunk_sum = sum(_tube_chunk_count(t) for t in state)

    hidden_unknowns = 0
    for t in state:
        # (맨 윗칸이 미지인 경우는 없다고 했지만, 혹시 생겨도 위로 노출되면 0으로 간주)
        for k, c in enumerate(t):
            if c.startswith("?") and k != 0:
                hidden_unknowns += 1

    empty = sum(1 for t in state if not t)

    tubes_by_color: Dict[str, int] = {}
    for t in state:
        if not t:
            continue
        for c in set(t):
            tubes_by_color[c] = tubes_by_color.get(c, 0) + 1
    separated = sum(1 for _, k in tubes_by_color.items() if k >= 2)

    bottoms = [t[-1] for t in state if t]
    distinct_bottoms = len(set(bottoms))

    return chunk_sum, hidden_unknowns, empty, separated, distinct_bottoms


def recommendation_score(state: State) -> tuple[int, int, int, int, int]:
    """
    lexicographic maximize를 위한 점수 튜플.
      - chunk_sum: 작을수록 좋음 => 음수로 변환
      - hidden_unknowns: 작을수록 좋음 => 음수로 변환
      - empty: 클수록 좋음
      - separated: 작을수록 좋음 => 음수로 변환
      - distinct_bottoms: 클수록 좋음
    """
    chunk_sum, hidden_unknowns, empty, separated, distinct_bottoms = recommendation_metrics(state)
    return (-chunk_sum, -hidden_unknowns, empty, -separated, distinct_bottoms)


def recommend_final_state(
    start_state: State,
    beam_width: int = 8000,
    max_steps: int = 350,
    time_limit_sec: float = 10.0,
) -> dict:
    """
    '가능한 상태들 중' 최종추천 점수(recommendation_score)가 가장 높은 상태를 찾는다.
    (휴리스틱 best-first + beam)
    """
    t0 = time.time()
    best_state = start_state
    best_sc = recommendation_score(start_state)

    def neg_sc(sc: tuple[int, int, int, int, int]) -> tuple[int, int, int, int, int]:
        return tuple(-x for x in sc)

    pq: list[tuple[tuple[int, int, int, int, int], int, State]] = []
    heapq.heappush(pq, (neg_sc(best_sc), 0, start_state))

    parent: Dict[State, Optional[State]] = {start_state: None}
    how: Dict[State, Optional[tuple[int, int, int, str]]] = {start_state: None}
    seen_best: Dict[State, tuple[tuple[int, int, int, int, int], int]] = {start_state: (best_sc, 0)}
    visited = 0

    while pq and (time.time() - t0) < time_limit_sec:
        _, steps, cur = heapq.heappop(pq)
        visited += 1

        cur_sc = recommendation_score(cur)
        if cur_sc > best_sc:
            best_sc = cur_sc
            best_state = cur

        if steps >= max_steps:
            continue

        nxt_candidates: list[tuple[tuple[int, int, int, int, int], int, int, int, str, State]] = []
        for i in range(len(cur)):
            for j in range(len(cur)):
                if i == j:
                    continue
                if not can_pour(cur[i], cur[j], i + 1, j + 1):
                    continue
                res = pour(cur, i, j)
                if res is None:
                    continue
                nxt, moved, color = res
                sc = recommendation_score(nxt)
                nxt_candidates.append((sc, i + 1, j + 1, moved, color, nxt))

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

    # reconstruct path to best_state
    path: List[tuple[int, int, int, str]] = []
    x = best_state
    while how.get(x) is not None:
        path.append(how[x])  # type: ignore[arg-type]
        x = parent[x]  # type: ignore[assignment]
    path.reverse()

    chunk_sum, hidden_unknowns, empty, separated, distinct_bottoms = recommendation_metrics(best_state)
    return {
        "visited": visited,
        "elapsed_sec": time.time() - t0,
        "best_state": best_state,
        "best_path": path,
        "metrics": {
            "chunk_sum": chunk_sum,
            "hidden_unknowns": hidden_unknowns,
            "empty_tubes": empty,
            "separated_colors": separated,
            "distinct_bottoms": distinct_bottoms,
        },
        "score": best_sc,
        "note": "최종추천 상태를 찾았습니다(휴리스틱).",
    }


def print_path(path: List[tuple[int, int, int, str]]) -> None:
    for k, (a, b, m, c) in enumerate(path, 1):
        print(f"{k:03d}. {a} -> {b} : {c} {m}")


def beam_solve(
    start_state: State,
    goal: str = "relaxed",
    beam_width: int = 6000,
    max_steps: int = 250,
    time_limit_sec: float = 12.0,
) -> dict:
    """
    해법 찾기(휴리스틱): best-first + beam.
    goal:
      - "strict": is_solved_strict
      - "relaxed": is_solved_relaxed
    """
    if goal == "strict":
        is_goal = is_solved_strict
    elif goal == "relaxed":
        is_goal = is_solved_relaxed
    else:
        raise ValueError(f"Unknown goal={goal}")

    t0 = time.time()
    best_state = start_state
    best_sc = score_relaxed(start_state)

    def neg_sc(sc: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(-x for x in sc)

    pq: list[tuple[tuple[int, int, int], int, State]] = []
    heapq.heappush(pq, (neg_sc(best_sc), 0, start_state))

    parent: Dict[State, Optional[State]] = {start_state: None}
    how: Dict[State, Optional[tuple[int, int, int, str]]] = {start_state: None}
    seen_best: Dict[State, tuple[tuple[int, int, int], int]] = {start_state: (best_sc, 0)}

    solved_state: Optional[State] = None
    visited = 0

    while pq and (time.time() - t0) < time_limit_sec:
        _, steps, cur = heapq.heappop(pq)
        visited += 1

        if is_goal(cur):
            solved_state = cur
            break

        cur_sc = score_relaxed(cur)
        if cur_sc > best_sc:
            best_sc = cur_sc
            best_state = cur

        if steps >= max_steps:
            continue

        nxt_candidates: list[tuple[tuple[int, int, int], int, int, int, str, State]] = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if not can_pour(cur[i], cur[j], i + 1, j + 1):
                    continue
                res = pour(cur, i, j)
                if res is None:
                    continue
                nxt, moved, color = res
                sc = score_relaxed(nxt)
                nxt_candidates.append((sc, i + 1, j + 1, moved, color, nxt))

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

    def reconstruct(end_state: State) -> List[tuple[int, int, int, str]]:
        path: List[tuple[int, int, int, str]] = []
        x = end_state
        while how.get(x) is not None:
            path.append(how[x])  # type: ignore[arg-type]
            x = parent[x]  # type: ignore[assignment]
        path.reverse()
        return path

    if solved_state is not None:
        return {
            "solved": True,
            "goal": goal,
            "visited": visited,
            "elapsed_sec": time.time() - t0,
            "state": solved_state,
            "path": reconstruct(solved_state),
            "best_state": solved_state,
            "best_score": score_relaxed(solved_state),
            "note": "해법을 찾았습니다(휴리스틱).",
        }

    return {
        "solved": False,
        "goal": goal,
        "visited": visited,
        "elapsed_sec": time.time() - t0,
        "state": None,
        "path": None,
        "best_state": best_state,
        "best_score": best_sc,
        "note": "시간/깊이/beam 제한 내에서는 해법을 찾지 못했습니다.",
    }


def bfs_solve(
    start_state: State,
    goal: str = "relaxed",
    max_nodes: int = 500_000,
) -> dict:
    """
    해법 찾기(BFS, 최단 경로).
    goal:
      - "strict": is_solved_strict
      - "relaxed": is_solved_relaxed
    """
    if goal == "strict":
        is_goal = is_solved_strict
    elif goal == "relaxed":
        is_goal = is_solved_relaxed
    else:
        raise ValueError(f"Unknown goal={goal}")

    if is_goal(start_state):
        return {
            "solved": True,
            "goal": goal,
            "visited": 1,
            "state": start_state,
            "path": [],
            "note": "이미 목표 상태입니다.",
        }

    q = deque([start_state])
    parent: Dict[State, Optional[State]] = {start_state: None}
    how: Dict[State, Optional[tuple[int, int, int, str]]] = {start_state: None}
    visited = 0

    while q and visited < max_nodes:
        cur = q.popleft()
        visited += 1

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if not can_pour(cur[i], cur[j], i + 1, j + 1):
                    continue
                res = pour(cur, i, j)
                if res is None:
                    continue
                nxt, moved, color = res
                if nxt in parent:
                    continue
                parent[nxt] = cur
                how[nxt] = (i + 1, j + 1, moved, color)
                if is_goal(nxt):
                    # reconstruct
                    path: List[tuple[int, int, int, str]] = []
                    x = nxt
                    while how[x] is not None:
                        path.append(how[x])  # type: ignore[arg-type]
                        x = parent[x]  # type: ignore[assignment]
                    path.reverse()
                    return {
                        "solved": True,
                        "goal": goal,
                        "visited": visited,
                        "state": nxt,
                        "path": path,
                        "note": "BFS로 해법(최단 경로)을 찾았습니다.",
                    }
                q.append(nxt)

    exhausted = not q
    note = (
        f"탐색 가능한 모든 상태(visited={visited}) 내에 해법이 없습니다."
        if exhausted
        else f"max_nodes={max_nodes} 내에서는 해법을 찾지 못했습니다. (visited={visited})"
    )
    return {
        "solved": False,
        "goal": goal,
        "visited": visited,
        "state": None,
        "path": None,
        "note": note,
    }


def reachable_make_empty_tube(
    tube_idx_1based: int,
    max_nodes: int = 5_000_000,
) -> dict:
    """
    목표: 특정 병(tube_idx_1based)을 '빈 병'으로 만들 수 있는지(BFS).

    반환 dict:
      {
        "possible": bool,
        "visited": int,
        "witness_state": State|None,
        "witness_path": List[(from,to,amount,color)]|None,
        "note": str
      }
    """
    if not (1 <= tube_idx_1based <= N):
        raise ValueError(f"tube_idx_1based must be in [1,{N}]")
    target0 = tube_idx_1based - 1

    def is_goal(state: State) -> bool:
        return len(state[target0]) == 0

    if is_goal(start):
        return {
            "possible": True,
            "visited": 1,
            "witness_state": start,
            "witness_path": [],
            "note": "이미 빈 병입니다.",
        }

    q = deque([start])
    parent: Dict[State, Optional[State]] = {start: None}
    how: Dict[State, Optional[tuple[int, int, int, str]]] = {start: None}
    visited = 0
    witness_state: Optional[State] = None

    while q and visited < max_nodes:
        cur = q.popleft()
        visited += 1

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if not can_pour(cur[i], cur[j], i + 1, j + 1):
                    continue
                res = pour(cur, i, j)
                if res is None:
                    continue
                nxt, moved, color = res
                if nxt in parent:
                    continue
                parent[nxt] = cur
                how[nxt] = (i + 1, j + 1, moved, color)
                if is_goal(nxt):
                    witness_state = nxt
                    q.clear()
                    break
                q.append(nxt)
            if witness_state is not None:
                break

    if witness_state is None:
        exhausted = not q
        note = (
            f"탐색 가능한 모든 상태(visited={visited}) 내에 목표가 없습니다."
            if exhausted
            else f"max_nodes={max_nodes} 내에서는 목표 달성 상태를 찾지 못했습니다. (visited={visited})"
        )
        return {
            "possible": False,
            "visited": visited,
            "witness_state": None,
            "witness_path": None,
            "note": note,
        }

    # 경로 복원
    path: List[tuple[int, int, int, str]] = []
    x = witness_state
    while how[x] is not None:
        path.append(how[x])
        x = parent[x]  # type: ignore
    path.reverse()

    return {
        "possible": True,
        "visited": visited,
        "witness_state": witness_state,
        "witness_path": path,
        "note": "목표 달성 상태를 찾았습니다.",
    }


def print_possible_colors_and_witnesses(
    goal_kind: str = "have_4_stack",
    max_nodes_per_color: int = 100_000_000,
) -> None:
    """
    start에 등장하는 모든 색(문자열 토큰 포함)을 대상으로,
    reachable_with_color로 목표 달성 가능 여부와 witness 경로를 출력.
    """
    all_colors = colors_in_state(start)
    possible_colors: List[str] = []
    print(f"\n=== 가능한 색 목록 (goal_kind={goal_kind}, max_nodes={max_nodes_per_color}) ===")
    for c in all_colors:
        r = reachable_with_color_multi(
            c,
            goal_kind=goal_kind,
            max_nodes=max_nodes_per_color,
            max_witnesses=1,
        )
        if r["possible"]:
            possible_colors.append(c)
            ws = r["witnesses"]
            print(f"\n- 색: {c}")
            print(f"  visited={r['visited']}, steps={len(ws[0]['path']) if ws else 0}")
            if ws and ws[0]["path"]:
                print("  witness moves:")
                print_path(ws[0]["path"])
    if not possible_colors:
        print("결과: 가능한 색을 찾지 못했습니다(탐색 한계일 수도 있습니다).")
    else:
        print(f"\n총 {len(possible_colors)}개 가능:", possible_colors)


def _task_color_witness(goal_kind: str, max_nodes_per_color: int, color: str) -> tuple[str, bool, int, list, bool, list, Optional[State]]:
    """Ray/병렬용: 단일 색에 대한 witness 계산."""
    r = reachable_with_color_multi(
        color,
        goal_kind=goal_kind,
        max_nodes=max_nodes_per_color,
        max_witnesses=1,
        stop_on_strict_solve=True,
    )
    if r.get("strict_solved"):
        return color, False, r["visited"], [], True, r["strict_path"], r["strict_state"]
    if not r["possible"]:
        return color, False, r["visited"], [], False, [], None
    ws = r["witnesses"]
    path = ws[0]["path"] if ws else []
    return color, True, r["visited"], path, False, [], None


def _task_empty_tube(tube_idx_1based: int, max_nodes_per_tube: int) -> tuple[int, bool, int, list]:
    """Ray/병렬용: 단일 병에 대한 '빈 병 만들기' witness 계산."""
    r = reachable_make_empty_tube(tube_idx_1based, max_nodes=max_nodes_per_tube)
    if not r["possible"]:
        return tube_idx_1based, False, r["visited"], []
    path = r["witness_path"] or []
    return tube_idx_1based, True, r["visited"], path


def print_possible_colors_and_witnesses_ray(
    goal_kind: str,
    max_nodes_per_color: int,
    max_procs: int = 8,
) -> None:
    """
    색별 탐색을 Ray로 병렬 실행.
    ray가 없으면 기존(순차) 출력으로 폴백.
    """
    try:
        import ray  # type: ignore
    except Exception:
        print_possible_colors_and_witnesses(goal_kind=goal_kind, max_nodes_per_color=max_nodes_per_color)
        return

    all_colors = colors_in_state(start)
    print(f"\n=== 가능한 색 목록 (goal_kind={goal_kind}, max_nodes={max_nodes_per_color}, ray={max_procs}) ===")

    ray.init(num_cpus=max_procs, ignore_reinit_error=True, include_dashboard=False)
    remote = ray.remote(_task_color_witness)
    refs = [remote.remote(goal_kind, max_nodes_per_color, c) for c in all_colors]

    possible_colors: List[str] = []
    results: List[tuple] = []
    strict_found: Optional[tuple[list, State]] = None
    pending = refs[:]
    try:
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            c, ok, visited, path, strict_solved, strict_path, strict_state = ray.get(ready[0])
            if strict_solved:
                strict_found = (strict_path, strict_state)
                # 남은 태스크는 더 진행할 필요 없음
                for rref in pending:
                    try:
                        ray.cancel(rref, force=True)
                    except Exception:
                        pass
                pending = []
                break
            results.append((c, ok, visited, path))
    finally:
        ray.shutdown()

    if strict_found is not None:
        spath, sstate = strict_found
        print("\n=== 엄격 완료 해법 발견 -> 탐색 중단 ===")
        print("이동 횟수:", len(spath))
        if spath:
            print_path(spath)
        print("\n=== 완료 상태(엄격) ===")
        print_state(sstate)
        return

    for c, ok, visited, path in sorted(results, key=lambda x: x[0]):
        if not ok:
            continue
        possible_colors.append(c)
        print(f"\n- 색: {c}")
        print(f"  visited={visited}, steps={len(path)}")
        if path:
            print("  witness moves:")
            print_path(path)

    if not possible_colors:
        print("결과: 가능한 색을 찾지 못했습니다(탐색 한계일 수도 있습니다).")
    else:
        print(f"\n총 {len(possible_colors)}개 가능:", possible_colors)


def print_empty_tube_combos_ray(max_nodes_per_tube: int = 5_000_000, max_procs: int = 8) -> None:
    """
    빈 병 만들기(병별)를 Ray로 병렬 실행.
    ray가 없으면 기존(순차) 출력으로 폴백.
    """
    try:
        import ray  # type: ignore
    except Exception:
        print_empty_tube_combos(max_nodes_per_tube=max_nodes_per_tube)
        return

    print(f"\n=== 빈 병 만들기 가능한 대상/경로 (max_nodes={max_nodes_per_tube}, ray={max_procs}) ===")

    ray.init(num_cpus=max_procs, ignore_reinit_error=True, include_dashboard=False)
    remote = ray.remote(_task_empty_tube)
    refs = [remote.remote(i, max_nodes_per_tube) for i in range(1, N + 1)]
    results = ray.get(refs)
    ray.shutdown()

    possible_targets: List[int] = []
    for i, ok, visited, path in sorted(results, key=lambda x: x[0]):
        if not ok:
            continue
        possible_targets.append(i)
        print(f"\n- 빈 병 대상: {i}번")
        print(f"  visited={visited}, steps={len(path)}")
        if path:
            print("  witness moves:")
            print_path(path)
        else:
            print("  (이미 빈 병)")

    if not possible_targets:
        print("결과: 빈 병을 만들 수 있는 대상을 찾지 못했습니다.")
    else:
        print(f"\n총 {len(possible_targets)}개 대상 가능:", possible_targets)


def print_empty_tube_combos(max_nodes_per_tube: int = 5_000_000) -> None:
    """
    '빈 병을 만들 수 있는 조합(이동 순서)' 출력:
    각 병 i에 대해 i를 빈 병으로 만들 수 있으면 witness 경로를 출력.
    """
    print(f"\n=== 빈 병 만들기 가능한 대상/경로 (max_nodes={max_nodes_per_tube}) ===")
    possible_targets: List[int] = []
    for i in range(1, N + 1):
        r = reachable_make_empty_tube(i, max_nodes=max_nodes_per_tube)
        if not r["possible"]:
            continue
        possible_targets.append(i)
        path = r["witness_path"] or []
        print(f"\n- 빈 병 대상: {i}번")
        print(f"  visited={r['visited']}, steps={len(path)}")
        if path:
            print("  witness moves:")
            print_path(path)
        else:
            print("  (이미 빈 병)")
    if not possible_targets:
        print("결과: 빈 병을 만들 수 있는 대상을 찾지 못했습니다.")
    else:
        print(f"\n총 {len(possible_targets)}개 대상 가능:", possible_targets)


def explain_strict_impossibility(state: State) -> Optional[str]:
    """
    엄격 완료(비었거나 4칸 단색)가 구조적으로 불가능한 경우를 빠르게 판정.
    - full 모드: 각 색의 개수는 0 또는 CAP(=4)이어야만 '완성 병'을 만들 수 있음.
    - partial 모드: 각 색의 개수는 CAP(=4) 이하여야 함. (이하일 수는 있음)
    """
    counts = color_counts(state)
    if COLOR_TOTAL_MODE == "full":
        bad = {c: n for c, n in counts.items() if n not in (0, CAP)}
    else:
        bad = {c: n for c, n in counts.items() if n > CAP}
    if not bad:
        return None
    parts = [f"{c}={n}" for c, n in sorted(bad.items(), key=lambda x: (-x[1], x[0]))]
    if COLOR_TOTAL_MODE == "full":
        return "엄격 완료는 색 개수 조건(각 색은 0 또는 4개) 때문에 불가능합니다: " + ", ".join(parts)
    return "엄격 완료는 색 개수 조건(각 색은 4개 이하) 때문에 불가능합니다: " + ", ".join(parts)


# =========================
# 사용 예시
# =========================
# 1) 특정 색이 4칸 단색 병으로 완성 가능한지:
# result = reachable_with_color("녹색", goal_kind="have_4_stack", max_nodes=300_000)
# print(result["possible"], result["visited"], result["note"])
# if result["possible"]:
#     for step in result["witness_path"]:
#         print(step)
#     print_state(result["witness_state"])

# 2) 특정 색이 맨 위로 올 수 있는지:
# result = reachable_with_color("보라", goal_kind="top_in_tube")

# 3) 특정 색이 한 번이라도 실제로 이동(붓기)되는지:
# result = reachable_with_color("연두", goal_kind="move_possible")


if __name__ == "__main__":
    try:
        print("=== 초기 상태 ===")
        print_state(start)
        print(f"\nCOLOR_TOTAL_MODE={COLOR_TOTAL_MODE} (full: 정확히 4개 / partial: 4개 이하)")
        print("\n초기 색 개수:", color_counts(start))

        # 너무 오래 걸리는 것을 방지하기 위해 기본 탐색 상한을 둡니다.
        # 필요하면 실행 시 환경변수로 올릴 수 있습니다.
        # 기본 탐색 상한(필요시 환경변수로 더 조절 가능)
        max_nodes_per_color = int(os.getenv("MAX_NODES_PER_COLOR", "3000000"))
        max_nodes_empty = int(os.getenv("MAX_NODES_EMPTY", "8000000"))
        max_nodes_two = int(os.getenv("MAX_NODES_TWO_COLORS", "12000000"))
        bfs_relaxed_max_nodes = int(os.getenv("BFS_RELAXED_MAX_NODES", "2000000"))
        rec_time = float(os.getenv("RECOMMEND_TIME", "20.0"))
        rec_beam = int(os.getenv("RECOMMEND_BEAM", "12000"))
        rec_steps = int(os.getenv("RECOMMEND_STEPS", "500"))
        max_procs = int(os.getenv("MAX_PROCS", "8"))

        # 1) 가능한 색 + witness 이동 모두 출력
        # print_possible_colors_and_witnesses(goal_kind="move_possible", max_nodes_per_color=max_nodes_per_color)
        print_possible_colors_and_witnesses_ray(goal_kind="have_4_stack", max_nodes_per_color=max_nodes_per_color, max_procs=max_procs)
        # print_empty_tube_combos_ray(max_nodes_per_tube=max_nodes_empty, max_procs=max_procs)

        # 1-b) 두 개의 색을 동시에 가능하게 하는 경로(한 경로로 4칸 단색 2개 동시 달성)
        print("\n=== 다색 동시 가능 경로 (full stack) ===")
        multi = reachable_multi_colors_have_full_stack(max_nodes=max_nodes_two, stop_on_strict_solve=True)
        if multi.get("strict_solved"):
            print(multi["note"], "visited=", multi["visited"])
            print("\n=== 엄격 완료 해법 발견 -> 탐색 중단 ===")
            spath = multi["strict_path"]
            print("이동 횟수:", len(spath))
            if spath:
                print_path(spath)
            print("\n=== 완료 상태(엄격) ===")
            print_state(multi["strict_state"])
            raise SystemExit(0)
        print(multi["note"], "visited=", multi["visited"])
        witness_by_k = multi["witness_by_k"]
        k = 2
        while k in witness_by_k:
            w = witness_by_k[k]
            print(f"\n--- {k}색 동시 ---")
            print("색:", ", ".join(w["colors"]))
            path = w["path"]
            print("이동 횟수:", len(path))
            if path:
                print_path(path)
            k += 1

        # 2) "모두 다 완성" 해법 존재 여부
        print("\n=== 전체 완료(엄격: 4칸 단색/빈 병) ===")
        msg = explain_strict_impossibility(start)
        if msg is not None:
            print(msg)
        else:
            res_strict = beam_solve(start, goal="strict", time_limit_sec=12.0)
            if res_strict["solved"]:
                print("해법을 찾았습니다.")
                print("이동 횟수:", len(res_strict["path"]))
                print_path(res_strict["path"])
                print("\n=== 완료 상태 ===")
                print_state(res_strict["state"])
            else:
                print(res_strict["note"])

        print("\n=== 전체 완료(완화: 단색/빈 병) ===")
        res = beam_solve(start, goal="relaxed", time_limit_sec=12.0, beam_width=8000, max_steps=300)
        print(f"visited={res['visited']}, elapsed={res['elapsed_sec']:.2f}s, best_score={res['best_score']}")
        if res["solved"]:
            print("해법을 찾았습니다.")
            print("이동 횟수:", len(res["path"]))
            print_path(res["path"])
            print("\n=== 완료 상태 ===")
            print_state(res["state"])
        else:
            print(res["note"])
            print("\n=== 현재까지 최고 상태(참고) ===")
            print_state(res["best_state"])

            # 휴리스틱이 실패하면 BFS도 한번 시도(상태 수 제한)
            print("\n--- BFS로 완화 완료 가능 여부 재확인(최단 경로) ---")
            bfs_res = bfs_solve(start, goal="relaxed", max_nodes=bfs_relaxed_max_nodes)
            print(bfs_res["note"], "visited=", bfs_res["visited"])
            if bfs_res["solved"]:
                print("이동 횟수:", len(bfs_res["path"]))
                print_path(bfs_res["path"])
                print("\n=== 완료 상태(BFS) ===")
                print_state(bfs_res["state"])

        # 3) 최종추천
        print("\n=== 최종추천(가장 유리한 상태) ===")
        rec = recommend_final_state(start, time_limit_sec=rec_time, beam_width=rec_beam, max_steps=rec_steps)
        m = rec["metrics"]
        print(
            f"visited={rec['visited']}, elapsed={rec['elapsed_sec']:.2f}s, "
            f"청크합={m['chunk_sum']}, 가려진미지={m['hidden_unknowns']}, 빈병={m['empty_tubes']}, "
            f"분리된색={m['separated_colors']}, 바닥색서로다름={m['distinct_bottoms']}"
        )
        path = rec["best_path"]
        print("이동 횟수:", len(path))
        if path:
            print_path(path)
        print("\n=== 최종추천 상태 ===")
        print_state(rec["best_state"])
    except BrokenPipeError:
        pass
