from __future__ import annotations
from collections import deque
import heapq
import time
import _engine as engine
import _scoring as scoring

CAP = 4
N = 10
TUBE_CAPS = [0] + [CAP] * N

# =========================
# "넣기" 제한 (목적지 병에 붓는 색 제한)
# 3번 병: 녹색만 넣기 가능
# =========================
restricted_put = {
    3: "녹색",
}

# =========================
# 초기 상태 (위 -> 아래)
# ? 3개는 서로 다른 미지색으로 둡니다: ?1 ?2 ?3
# =========================
start = (
    ("노랑","노랑","빨강","연두"),                 # 1
    ("빨강","?1","?2","?3"),                        # 2
    ("하늘","연두","보라","녹색"),                 # 3  (녹색만 넣기 가능)
    ("회색",),                                      # 4
    ("회색","하늘","하늘","녹색"),                 # 5
    ("자색","자색","주황"),                         # 6
    ("녹색",),                                      # 7
    ("주황","주황","하늘","연두"),                 # 8
    ("회색","빨강","빨강"),                         # 9
    ("갈색","노랑","핑크","갈색"),                 # 10
)

# =========================
# 규칙 함수
# =========================
def can_put(dst_idx_1based: int, color: str) -> bool:
    return engine.can_put(restricted_put, dst_idx_1based, color)

def can_pour(src: tuple[str,...], dst: tuple[str,...], src_i: int, dst_i: int) -> bool:
    return engine.can_pour(TUBE_CAPS, restricted_put, set(), set(), src, dst, src_i, dst_i)

def pour(state: tuple[tuple[str,...],...], i0: int, j0: int):
    """0-based i0->j0, 가능한 만큼 연속 붓기"""
    return engine.pour(TUBE_CAPS, state, i0, j0)

# =========================
# "최대한 많이 모으기" 목적 함수(점수)
# 1) 완성된 단색 4칸 병 개수 (가장 중요)
# 2) 단색(비었거나 한 색만) 병 개수
# 3) 각 병에서 가장 많은 동일색 개수 합(응집도)
# =========================
def score(state):
    return scoring.score_maximize(state, tube_caps_1based=TUBE_CAPS)

# =========================
# 탐색: Best-first + Beam 제한
# - beam_width: 한 "깊이"에서 유지할 후보 수
# - max_steps: 경로 길이 제한
# - time_limit_sec: 시간 제한
# =========================
def search_maximize(start, beam_width=4000, max_steps=160, time_limit_sec=10.0):
    t0 = time.time()

    best_state = start
    best_path = []
    best_score = score(start)

    # 우선순위 큐: ( -score, steps, state )
    # Python heap은 min-heap이라 score는 음수로
    def neg_sc(sc):  # 튜플에 음수
        return tuple(-x for x in sc)

    pq = []
    heapq.heappush(pq, (neg_sc(best_score), 0, start))

    # 방문 체크(너무 커지면 성능 떨어져서, 점수+깊이 기반으로 느슨하게)
    seen = {start: (best_score, 0)}
    parent = {start: None}
    how = {start: None}

    while pq and (time.time() - t0) < time_limit_sec:
        _, steps, cur = heapq.heappop(pq)

        cur_sc = score(cur)
        if cur_sc > best_score:
            best_score = cur_sc
            best_state = cur
            # 경로 복원
            path = []
            x = cur
            while how[x] is not None:
                path.append(how[x])
                x = parent[x]
            path.reverse()
            best_path = path

        if steps >= max_steps:
            continue

        # 다음 상태 생성
        nxt_candidates = []
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
                sc = score(nxt)
                nxt_candidates.append((sc, i+1, j+1, moved, color, nxt))

        # 후보가 많으면 점수 상위만 beam_width만큼
        nxt_candidates.sort(key=lambda x: x[0], reverse=True)
        if len(nxt_candidates) > beam_width:
            nxt_candidates = nxt_candidates[:beam_width]

        for sc, a, b, m, c, nxt in nxt_candidates:
            prev = seen.get(nxt)
            # 더 좋은 점수/더 짧은 경로면 갱신
            if (prev is None) or (sc > prev[0]) or (sc == prev[0] and steps+1 < prev[1]):
                seen[nxt] = (sc, steps+1)
                parent[nxt] = cur
                how[nxt] = (a, b, m, c)  # 1-based
                heapq.heappush(pq, (neg_sc(sc), steps+1, nxt))

    return best_state, best_path, best_score, time.time() - t0

# =========================
# 출력
# =========================
def fmt_tube(t):
    return "[" + ", ".join(t) + "]"

def print_state(state):
    # maximize.py는 (고) 개념을 사용하지 않으므로 locked_out은 빈 set
    engine.print_state(state, restricted_put, set(), no_put=set(), tube_caps_1based=TUBE_CAPS)

best_state, best_path, best_sc, elapsed = search_maximize(
    start,
    beam_width=5000,
    max_steps=200,
    time_limit_sec=12.0,
)

print("=== 초기 상태 ===")
print_state(start)
print("\n초기 점수:", score(start))

print(f"\n=== 탐색 결과 (경과 {elapsed:.2f}s) ===")
print("최고 점수:", best_sc)
print("이동 횟수:", len(best_path))

if best_path:
    print("\n--- 추천 이동 순서 ---")
    for k, (a,b,m,c) in enumerate(best_path, 1):
        print(f"{k:03d}. {a} -> {b} : {c} {m}")

print("\n=== 결과 상태 ===")
print_state(best_state)

# 참고: "완성된 4칸 단색" 병만 따로 보여주기
complete = []
for i, t in enumerate(best_state, 1):
    if len(t) == CAP and len(set(t)) == 1:
        complete.append((i, t[0]))
print("\n완성된 단색 4칸 병:", complete if complete else "없음")
