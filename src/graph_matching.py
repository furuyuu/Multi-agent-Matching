from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except ModuleNotFoundError:
    _scipy_linear_sum_assignment = None

from src.simulation import Detection, wrap_angle


ROCO_PARAMS = {
    "tau2": 6.0,
    "tau1": 0.3,
    "lambda_dist": 1.0,
    "neighbor_radius": 15.0,
}

RRWM_PARAMS = {
    "candidate_radius": 6.0,
    "unary_weight": 2.0,
    "pairwise_weight": 1.0,
    "sigma_pos": 2.0,
    "sigma_yaw_deg": 15.0,
    "sigma_edge": 1.5,
    "max_iter": 10000,
    "alpha": 0.2,
    "beta": 30.0,
    "score_threshold": 0.02,
}

KWISE_PARAMS = RRWM_PARAMS.copy()


# ============================================================
# Shared utilities
# ============================================================


def linear_sum_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hungarian assignment を解きます。

    scipy が使える環境では scipy.optimize.linear_sum_assignment を使います。
    scipy がない環境では、小規模実験用の動的計画法フォールバックを使います。
    """

    if _scipy_linear_sum_assignment is not None:
        return _scipy_linear_sum_assignment(cost_matrix)

    cost_matrix = np.asarray(cost_matrix)
    num_rows, num_cols = cost_matrix.shape

    if num_rows == 0 or num_cols == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # DP は「行数 <= 列数」の形で解くと扱いやすいため、必要なら転置します。
    transposed = False
    work_cost = cost_matrix
    if num_rows > num_cols:
        work_cost = cost_matrix.T
        num_rows, num_cols = work_cost.shape
        transposed = True

    # dp[mask] = ここまでの行を、mask で表される列集合へ割り当てたときの最小コスト
    dp = {0: (0.0, [])}
    for row in range(num_rows):
        next_dp = {}
        for mask, (current_cost, cols) in dp.items():
            for col in range(num_cols):
                if mask & (1 << col):
                    continue
                next_mask = mask | (1 << col)
                next_cost = current_cost + work_cost[row, col]
                if next_mask not in next_dp or next_cost < next_dp[next_mask][0]:
                    next_dp[next_mask] = (next_cost, cols + [col])
        dp = next_dp

    _, assigned_cols = min(dp.values(), key=lambda item: item[0])
    row_ind = np.arange(num_rows, dtype=int)
    col_ind = np.array(assigned_cols, dtype=int)

    if transposed:
        return col_ind, row_ind
    return row_ind, col_ind


def detection_position(det: Detection) -> np.ndarray:
    """検出物体の global 推定位置を2次元ベクトルとして返します。"""

    return np.array([det.x_global_est, det.y_global_est])


def distance_between_detections(det_p: Detection, det_q: Detection) -> float:
    """2つの検出物体間の global 座標上のユークリッド距離を計算します。"""

    return float(np.linalg.norm(detection_position(det_p) - detection_position(det_q)))


def pose_matrix_from_detection(det: Detection) -> np.ndarray:
    """Detection の global 推定 pose から 2D 同次変換行列を作ります。"""

    x = det.x_global_est
    y = det.y_global_est
    theta = det.theta_global_est
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array(
        [
            [c, -s, x],
            [s, c, y],
            [0, 0, 1],
        ]
    )


def relative_pose_transform(center_det: Detection, neighbor_det: Detection) -> np.ndarray:
    """center_det から neighbor_det への相対 pose 変換行列を計算します。"""

    t_center = pose_matrix_from_detection(center_det)
    t_neighbor = pose_matrix_from_detection(neighbor_det)
    return np.linalg.inv(t_center) @ t_neighbor


def stable_softmax_like(x: np.ndarray, beta: float) -> np.ndarray:
    """RRWM の exponential reweighting を数値安定に計算します。"""

    z = beta * x
    z = z - np.max(z)
    y = np.exp(z)
    return y / np.maximum(np.sum(y), 1e-12)


# ============================================================
# 1. RoCo-style object matching
# ============================================================


def edge_consistency_score(t_pm: np.ndarray, t_qn: np.ndarray) -> float:
    """RoCo の edge consistency に相当するスコアを計算します。"""

    diff = t_pm @ np.linalg.inv(t_qn) - np.eye(3)
    return float(np.exp(-np.linalg.norm(diff, ord="fro")))


def get_neighbors(
    center_det: Detection,
    detections: List[Detection],
    neighbor_radius: float = 12.0,
) -> List[Detection]:
    """center_det の周辺物体を取得します。"""

    neighbors = []
    for det in detections:
        if det.det_id == center_det.det_id:
            continue
        if distance_between_detections(center_det, det) <= neighbor_radius:
            neighbors.append(det)
    return neighbors


def initial_distance_matching(
    dets_i: List[Detection],
    dets_j: List[Detection],
    tau2: float = 4.0,
) -> Dict[int, int]:
    """距離に基づき、各 det_i に最も近い det_j を初期対応として選びます。"""

    initial_match = {}
    for p in dets_i:
        best_q = None
        best_dist = np.inf
        for q in dets_j:
            dist = distance_between_detections(p, q)
            if dist <= tau2 and dist < best_dist:
                best_dist = dist
                best_q = q
        if best_q is not None:
            initial_match[p.det_id] = best_q.det_id
    return initial_match


def edge_similarity(
    p: Detection,
    q: Detection,
    dets_i: List[Detection],
    dets_j: List[Detection],
    initial_match_i_to_j: Dict[int, int],
    neighbor_radius: float = 12.0,
) -> float:
    """p と q の周辺構造がどれだけ似ているかを評価します。"""

    neighbors_p = get_neighbors(p, dets_i, neighbor_radius=neighbor_radius)
    if len(neighbors_p) == 0:
        return 0.0

    dets_j_dict = {det.det_id: det for det in dets_j}
    scores = []

    for m in neighbors_p:
        if m.det_id not in initial_match_i_to_j:
            continue

        n_id = initial_match_i_to_j[m.det_id]
        if n_id not in dets_j_dict:
            continue

        n = dets_j_dict[n_id]
        t_pm = relative_pose_transform(p, m)
        t_qn = relative_pose_transform(q, n)
        scores.append(edge_consistency_score(t_pm, t_qn))

    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores))


def distance_similarity(p: Detection, q: Detection) -> float:
    """global 座標上で近いほど高くなる距離 similarity を返します。"""

    return float(np.exp(-distance_between_detections(p, q)))


def roco_similarity_matrix(
    dets_i: List[Detection],
    dets_j: List[Detection],
    tau2: float = 4.0,
    tau1: float = 2.5,
    lambda_dist: float = 1.0,
    neighbor_radius: float = 12.0,
) -> np.ndarray:
    """agent i と agent j の検出物体間の RoCo 風 similarity matrix を作ります。"""

    initial_match_i_to_j = initial_distance_matching(dets_i, dets_j, tau2=tau2)
    s_mat = np.zeros((len(dets_i), len(dets_j)))

    for a, p in enumerate(dets_i):
        for b, q in enumerate(dets_j):
            if distance_between_detections(p, q) > tau2:
                continue

            s_edge = edge_similarity(
                p,
                q,
                dets_i,
                dets_j,
                initial_match_i_to_j,
                neighbor_radius=neighbor_radius,
            )
            s_dist = distance_similarity(p, q)
            score = s_edge + lambda_dist * s_dist
            s_mat[a, b] = score if score >= tau1 else 0.0

    return s_mat


def roco_pairwise_matching(
    dets_i: List[Detection],
    dets_j: List[Detection],
    tau2: float = 4.0,
    tau1: float = 0.5,
    lambda_dist: float = 1.0,
    neighbor_radius: float = 12.0,
) -> List[Tuple[int, int, float]]:
    """RoCo 風 similarity を作り、Hungarian algorithm で1対1対応を求めます。"""

    s_mat = roco_similarity_matrix(
        dets_i,
        dets_j,
        tau2=tau2,
        tau1=tau1,
        lambda_dist=lambda_dist,
        neighbor_radius=neighbor_radius,
    )
    if np.all(s_mat == 0):
        return []

    row_ind, col_ind = linear_sum_assignment(-s_mat)
    matches = []
    for r, c in zip(row_ind, col_ind):
        score = s_mat[r, c]
        if score > 0:
            matches.append((dets_i[r].det_id, dets_j[c].det_id, float(score)))
    return matches


# ============================================================
# 2. Pairwise graph matching with RRWM
# ============================================================


def build_candidate_matches(
    dets_i: List[Detection],
    dets_j: List[Detection],
    candidate_radius: float = 8.0,
) -> List[Tuple[int, int]]:
    """pairwise association graph の候補ノードを作ります。"""

    candidates = []
    for a, di in enumerate(dets_i):
        for b, dj in enumerate(dets_j):
            if distance_between_detections(di, dj) <= candidate_radius:
                candidates.append((a, b))
    return candidates


def unary_affinity(
    di: Detection,
    dj: Detection,
    sigma_pos: float = 3.0,
    sigma_yaw_deg: float = 20.0,
) -> float:
    """位置差と向き差から、候補対応そのものの類似度を計算します。"""

    pos_dist = distance_between_detections(di, dj)
    yaw_diff = abs(wrap_angle(di.theta_global_est - dj.theta_global_est))
    sigma_yaw = np.deg2rad(sigma_yaw_deg)
    score_pos = np.exp(-(pos_dist**2) / (2 * sigma_pos**2))
    score_yaw = np.exp(-(yaw_diff**2) / (2 * sigma_yaw**2))
    return float(score_pos * score_yaw)


def pairwise_edge_affinity(
    di_a: Detection,
    di_c: Detection,
    dj_b: Detection,
    dj_d: Detection,
    sigma_edge: float = 3.0,
) -> float:
    """2つの対応仮説が同時に成立しやすいかを、相対 pose の整合性で評価します。"""

    t_ac = relative_pose_transform(di_a, di_c)
    t_bd = relative_pose_transform(dj_b, dj_d)
    error = np.linalg.norm(t_ac @ np.linalg.inv(t_bd) - np.eye(3), ord="fro")
    return float(np.exp(-(error**2) / (2 * sigma_edge**2)))


def build_affinity_matrix_pairwise(
    dets_i: List[Detection],
    dets_j: List[Detection],
    candidates: List[Tuple[int, int]],
    unary_weight: float = 1.0,
    pairwise_weight: float = 1.0,
    sigma_pos: float = 3.0,
    sigma_yaw_deg: float = 20.0,
    sigma_edge: float = 3.0,
) -> np.ndarray:
    """pairwise association graph の affinity matrix を作ります。"""

    n = len(candidates)
    k_mat = np.zeros((n, n))

    for u, (a, b) in enumerate(candidates):
        di_a = dets_i[a]
        dj_b = dets_j[b]
        k_mat[u, u] = unary_weight * unary_affinity(
            di_a,
            dj_b,
            sigma_pos=sigma_pos,
            sigma_yaw_deg=sigma_yaw_deg,
        )

        for v, (c, d) in enumerate(candidates):
            if u == v or a == c or b == d:
                continue

            k_mat[u, v] = pairwise_weight * pairwise_edge_affinity(
                di_a,
                dets_i[c],
                dj_b,
                dets_j[d],
                sigma_edge=sigma_edge,
            )

    return k_mat


def normalize_by_groups(
    x: np.ndarray,
    candidates: List[Tuple[int, int]],
    num_i: int,
    num_j: int,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """候補スコアを agent i 側・agent j 側のグループごとに正規化します。"""

    x = x.copy()
    eps = 1e-12

    for _ in range(max_iter):
        x_prev = x.copy()

        for a in range(num_i):
            idx = [k for k, (i_idx, _) in enumerate(candidates) if i_idx == a]
            s = np.sum(x[idx])
            if s > eps:
                x[idx] /= s

        for b in range(num_j):
            idx = [k for k, (_, j_idx) in enumerate(candidates) if j_idx == b]
            s = np.sum(x[idx])
            if s > eps:
                x[idx] /= s

        if np.linalg.norm(x - x_prev) < tol:
            break

    total = np.sum(x)
    if total > eps:
        x /= total
    return x


def rrwm(
    k_mat: np.ndarray,
    candidates: List[Tuple[int, int]],
    num_i: int,
    num_j: int,
    max_iter: int = 100,
    alpha: float = 0.2,
    beta: float = 30.0,
    tol: float = 1e-6,
) -> np.ndarray:
    """pairwise 候補に対する RRWM 最適化を行います。"""

    n = k_mat.shape[0]
    if n == 0:
        return np.array([])

    col_sum = k_mat.sum(axis=0, keepdims=True)
    k_norm = k_mat / np.maximum(col_sum, 1e-12)
    x = np.ones(n) / n

    for _ in range(max_iter):
        x_old = x.copy()
        x = k_norm @ x
        x = stable_softmax_like(x, beta=beta)
        x = normalize_by_groups(x, candidates, num_i=num_i, num_j=num_j)
        x = alpha * x_old + (1 - alpha) * x
        x = x / np.maximum(np.sum(x), 1e-12)
        if np.linalg.norm(x - x_old) < tol:
            break

    return x


def discretize_rrwm_solution(
    x: np.ndarray,
    dets_i: List[Detection],
    dets_j: List[Detection],
    candidates: List[Tuple[int, int]],
    score_threshold: float = 1e-4,
) -> List[Tuple[int, int, float]]:
    """RRWM の soft score を Hungarian algorithm で1対1対応に変換します。"""

    if len(candidates) == 0:
        return []

    score_matrix = np.zeros((len(dets_i), len(dets_j)))
    for k, (a, b) in enumerate(candidates):
        score_matrix[a, b] = x[k]

    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    matches = []
    for r, c in zip(row_ind, col_ind):
        score = score_matrix[r, c]
        if score > score_threshold:
            matches.append((dets_i[r].det_id, dets_j[c].det_id, float(score)))
    return matches


def pairwise_rrwm_matching(
    dets_i: List[Detection],
    dets_j: List[Detection],
    candidate_radius: float = 8.0,
    unary_weight: float = 1.0,
    pairwise_weight: float = 1.0,
    sigma_pos: float = 3.0,
    sigma_yaw_deg: float = 20.0,
    sigma_edge: float = 3.0,
    max_iter: int = 100,
    alpha: float = 0.2,
    beta: float = 30.0,
    score_threshold: float = 1e-4,
) -> Tuple[List[Tuple[int, int, float]], np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """pairwise graph matching + RRWM の全体関数です。"""

    candidates = build_candidate_matches(
        dets_i,
        dets_j,
        candidate_radius=candidate_radius,
    )
    k_mat = build_affinity_matrix_pairwise(
        dets_i,
        dets_j,
        candidates,
        unary_weight=unary_weight,
        pairwise_weight=pairwise_weight,
        sigma_pos=sigma_pos,
        sigma_yaw_deg=sigma_yaw_deg,
        sigma_edge=sigma_edge,
    )
    x = rrwm(
        k_mat,
        candidates,
        num_i=len(dets_i),
        num_j=len(dets_j),
        max_iter=max_iter,
        alpha=alpha,
        beta=beta,
    )
    matches = discretize_rrwm_solution(
        x,
        dets_i,
        dets_j,
        candidates,
        score_threshold=score_threshold,
    )
    return matches, x, k_mat, candidates


# ============================================================
# 3. 3-wise matching with RRWM
# ============================================================


def build_kwise_candidates_3agents(
    dets_0: List[Detection],
    dets_1: List[Detection],
    dets_2: List[Detection],
    candidate_radius: float = 8.0,
) -> List[Tuple[int, int, int]]:
    """3エージェント間の三つ組候補を作ります。"""

    candidates = []
    for a, d0 in enumerate(dets_0):
        for b, d1 in enumerate(dets_1):
            for c, d2 in enumerate(dets_2):
                d01 = distance_between_detections(d0, d1)
                d02 = distance_between_detections(d0, d2)
                d12 = distance_between_detections(d1, d2)
                if d01 <= candidate_radius and d02 <= candidate_radius and d12 <= candidate_radius:
                    candidates.append((a, b, c))
    return candidates


def kwise_unary_affinity(
    d0: Detection,
    d1: Detection,
    d2: Detection,
    sigma_pos: float = 3.0,
    sigma_yaw_deg: float = 20.0,
) -> float:
    """三つ組対応そのものの類似度を、位置差と向き差から計算します。"""

    pos01 = distance_between_detections(d0, d1)
    pos02 = distance_between_detections(d0, d2)
    pos12 = distance_between_detections(d1, d2)
    yaw01 = abs(wrap_angle(d0.theta_global_est - d1.theta_global_est))
    yaw02 = abs(wrap_angle(d0.theta_global_est - d2.theta_global_est))
    yaw12 = abs(wrap_angle(d1.theta_global_est - d2.theta_global_est))

    sigma_yaw = np.deg2rad(sigma_yaw_deg)
    pos_error = pos01**2 + pos02**2 + pos12**2
    yaw_error = yaw01**2 + yaw02**2 + yaw12**2
    score_pos = np.exp(-pos_error / (2 * sigma_pos**2))
    score_yaw = np.exp(-yaw_error / (2 * sigma_yaw**2))
    return float(score_pos * score_yaw)


def kwise_pairwise_structure_affinity(
    d0_a: Detection,
    d1_b: Detection,
    d2_c: Detection,
    d0_p: Detection,
    d1_q: Detection,
    d2_r: Detection,
    sigma_edge: float = 3.0,
) -> float:
    """2つの三つ組候補の構造整合性を min-based affinity で評価します。"""

    t0 = relative_pose_transform(d0_a, d0_p)
    t1 = relative_pose_transform(d1_b, d1_q)
    t2 = relative_pose_transform(d2_c, d2_r)

    err01 = np.linalg.norm(t0 @ np.linalg.inv(t1) - np.eye(3), ord="fro")
    err02 = np.linalg.norm(t0 @ np.linalg.inv(t2) - np.eye(3), ord="fro")
    err12 = np.linalg.norm(t1 @ np.linalg.inv(t2) - np.eye(3), ord="fro")
    s01 = np.exp(-(err01**2) / (2 * sigma_edge**2))
    s02 = np.exp(-(err02**2) / (2 * sigma_edge**2))
    s12 = np.exp(-(err12**2) / (2 * sigma_edge**2))
    return float(min(s01, s02, s12))


def build_affinity_matrix_kwise_3agents(
    dets_0: List[Detection],
    dets_1: List[Detection],
    dets_2: List[Detection],
    candidates: List[Tuple[int, int, int]],
    unary_weight: float = 1.0,
    pairwise_weight: float = 1.0,
    sigma_pos: float = 3.0,
    sigma_yaw_deg: float = 20.0,
    sigma_edge: float = 3.0,
) -> np.ndarray:
    """三つ組候補をノードとする k-wise affinity matrix を作ります。"""

    n = len(candidates)
    k_mat = np.zeros((n, n))

    for u, (a, b, c) in enumerate(candidates):
        d0_a = dets_0[a]
        d1_b = dets_1[b]
        d2_c = dets_2[c]
        k_mat[u, u] = unary_weight * kwise_unary_affinity(
            d0_a,
            d1_b,
            d2_c,
            sigma_pos=sigma_pos,
            sigma_yaw_deg=sigma_yaw_deg,
        )

        for v, (p, q, r) in enumerate(candidates):
            if u == v or a == p or b == q or c == r:
                continue

            k_mat[u, v] = pairwise_weight * kwise_pairwise_structure_affinity(
                d0_a,
                d1_b,
                d2_c,
                dets_0[p],
                dets_1[q],
                dets_2[r],
                sigma_edge=sigma_edge,
            )

    return k_mat


def normalize_by_groups_kwise_3agents(
    x: np.ndarray,
    candidates: List[Tuple[int, int, int]],
    num_0: int,
    num_1: int,
    num_2: int,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """3エージェント版の group normalization を行います。"""

    x = x.copy()
    eps = 1e-12

    for _ in range(max_iter):
        x_prev = x.copy()

        for a in range(num_0):
            idx = [k for k, (i0, _, _) in enumerate(candidates) if i0 == a]
            s = np.sum(x[idx])
            if s > eps:
                x[idx] /= s

        for b in range(num_1):
            idx = [k for k, (_, i1, _) in enumerate(candidates) if i1 == b]
            s = np.sum(x[idx])
            if s > eps:
                x[idx] /= s

        for c in range(num_2):
            idx = [k for k, (_, _, i2) in enumerate(candidates) if i2 == c]
            s = np.sum(x[idx])
            if s > eps:
                x[idx] /= s

        if np.linalg.norm(x - x_prev) < tol:
            break

    total = np.sum(x)
    if total > eps:
        x /= total
    return x


def rrwm_kwise_3agents(
    k_mat: np.ndarray,
    candidates: List[Tuple[int, int, int]],
    num_0: int,
    num_1: int,
    num_2: int,
    max_iter: int = 100,
    alpha: float = 0.2,
    beta: float = 30.0,
    tol: float = 1e-6,
) -> np.ndarray:
    """3エージェント k-wise 候補に対する RRWM を行います。"""

    n = k_mat.shape[0]
    if n == 0:
        return np.array([])

    col_sum = k_mat.sum(axis=0, keepdims=True)
    k_norm = k_mat / np.maximum(col_sum, 1e-12)
    x = np.ones(n) / n

    for _ in range(max_iter):
        x_old = x.copy()
        x = k_norm @ x
        x = stable_softmax_like(x, beta=beta)
        x = normalize_by_groups_kwise_3agents(
            x,
            candidates,
            num_0=num_0,
            num_1=num_1,
            num_2=num_2,
        )
        x = alpha * x_old + (1 - alpha) * x
        x = x / np.maximum(np.sum(x), 1e-12)
        if np.linalg.norm(x - x_old) < tol:
            break

    return x


def discretize_kwise_solution_greedy(
    x: np.ndarray,
    dets_0: List[Detection],
    dets_1: List[Detection],
    dets_2: List[Detection],
    candidates: List[Tuple[int, int, int]],
    score_threshold: float = 1e-4,
) -> List[Tuple[int, int, int, float]]:
    """RRWM の soft score を greedy に離散的な3エージェント対応へ変換します。"""

    if len(candidates) == 0:
        return []

    order = np.argsort(-x)
    used_0 = set()
    used_1 = set()
    used_2 = set()
    matches = []

    for k in order:
        score = x[k]
        if score < score_threshold:
            continue

        a, b, c = candidates[k]
        if a in used_0 or b in used_1 or c in used_2:
            continue

        used_0.add(a)
        used_1.add(b)
        used_2.add(c)
        matches.append((dets_0[a].det_id, dets_1[b].det_id, dets_2[c].det_id, float(score)))

    return matches


def kwise_rrwm_matching_3agents(
    dets_0: List[Detection],
    dets_1: List[Detection],
    dets_2: List[Detection],
    candidate_radius: float = 8.0,
    unary_weight: float = 1.0,
    pairwise_weight: float = 1.0,
    sigma_pos: float = 3.0,
    sigma_yaw_deg: float = 20.0,
    sigma_edge: float = 3.0,
    max_iter: int = 100,
    alpha: float = 0.2,
    beta: float = 30.0,
    score_threshold: float = 1e-4,
) -> Tuple[
    List[Tuple[int, int, int, float]],
    np.ndarray,
    np.ndarray,
    List[Tuple[int, int, int]],
]:
    """k-wise MGM style + RRWM の全体関数です。"""

    candidates = build_kwise_candidates_3agents(
        dets_0,
        dets_1,
        dets_2,
        candidate_radius=candidate_radius,
    )
    k_mat = build_affinity_matrix_kwise_3agents(
        dets_0,
        dets_1,
        dets_2,
        candidates,
        unary_weight=unary_weight,
        pairwise_weight=pairwise_weight,
        sigma_pos=sigma_pos,
        sigma_yaw_deg=sigma_yaw_deg,
        sigma_edge=sigma_edge,
    )
    x = rrwm_kwise_3agents(
        k_mat,
        candidates,
        num_0=len(dets_0),
        num_1=len(dets_1),
        num_2=len(dets_2),
        max_iter=max_iter,
        alpha=alpha,
        beta=beta,
    )
    matches = discretize_kwise_solution_greedy(
        x,
        dets_0,
        dets_1,
        dets_2,
        candidates,
        score_threshold=score_threshold,
    )
    return matches, x, k_mat, candidates


# ============================================================
# Evaluation helpers
# ============================================================


def evaluate_pairwise_matches(
    dets_i: List[Detection],
    dets_j: List[Detection],
    matches: List[Tuple[int, int, float]],
) -> Dict[str, float]:
    """pairwise matching の precision / recall / F1 を計算します。"""

    dets_i_by_id = {det.det_id: det for det in dets_i}
    dets_j_by_id = {det.det_id: det for det in dets_j}
    true_ids_i = {det.true_obj_id for det in dets_i if det.true_obj_id >= 0}
    true_ids_j = {det.true_obj_id for det in dets_j if det.true_obj_id >= 0}
    num_gt = len(true_ids_i & true_ids_j)

    tp = 0
    fp = 0
    for det_i_id, det_j_id, _ in matches:
        det_i = dets_i_by_id[det_i_id]
        det_j = dets_j_by_id[det_j_id]
        if det_i.true_obj_id >= 0 and det_i.true_obj_id == det_j.true_obj_id:
            tp += 1
        else:
            fp += 1

    fn = max(num_gt - tp, 0)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "num_gt": float(num_gt),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_pairwise_matching(
    dets_i: List[Detection],
    dets_j: List[Detection],
    matches: List[Tuple[int, int, float]],
) -> Dict[str, float]:
    """pairwise matching を notebook 表示向けの項目名で評価します。"""

    metrics = evaluate_pairwise_matches(dets_i, dets_j, matches)
    correct = int(metrics["tp"])
    predicted_matches = len(matches)
    gt_matches = int(metrics["num_gt"])

    return {
        "correct": correct,
        "predicted_matches": predicted_matches,
        "gt_matches": gt_matches,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
    }


def evaluate_kwise_matches_3agents(
    dets_0: List[Detection],
    dets_1: List[Detection],
    dets_2: List[Detection],
    matches: List[Tuple[int, int, int, float]],
) -> Dict[str, float]:
    """3-wise matching の precision / recall / F1 を計算します。"""

    dets_0_by_id = {det.det_id: det for det in dets_0}
    dets_1_by_id = {det.det_id: det for det in dets_1}
    dets_2_by_id = {det.det_id: det for det in dets_2}
    true_ids_0 = {det.true_obj_id for det in dets_0 if det.true_obj_id >= 0}
    true_ids_1 = {det.true_obj_id for det in dets_1 if det.true_obj_id >= 0}
    true_ids_2 = {det.true_obj_id for det in dets_2 if det.true_obj_id >= 0}
    num_gt = len(true_ids_0 & true_ids_1 & true_ids_2)

    tp = 0
    fp = 0
    for det_0_id, det_1_id, det_2_id, _ in matches:
        det_0 = dets_0_by_id[det_0_id]
        det_1 = dets_1_by_id[det_1_id]
        det_2 = dets_2_by_id[det_2_id]
        same_true_object = (
            det_0.true_obj_id >= 0
            and det_0.true_obj_id == det_1.true_obj_id
            and det_0.true_obj_id == det_2.true_obj_id
        )
        if same_true_object:
            tp += 1
        else:
            fp += 1

    fn = max(num_gt - tp, 0)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "num_gt": float(num_gt),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_kwise_matching_3agents(
    dets_0: List[Detection],
    dets_1: List[Detection],
    dets_2: List[Detection],
    matches: List[Tuple[int, int, int, float]],
) -> Dict[str, float]:
    """3エージェント同時対応を notebook 表示向けの項目名で評価します。"""

    metrics = evaluate_kwise_matches_3agents(dets_0, dets_1, dets_2, matches)
    return {
        "TP_correct_matches": int(metrics["tp"]),
        "FP_wrong_matches": int(metrics["fp"]),
        "FN_missed_matches": int(metrics["fn"]),
        "gt_matches": int(metrics["num_gt"]),
        "predicted_matches": len(matches),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
    }


def kwise_to_pairwise_matches(
    kwise_matches: List[Tuple[int, int, int, float]],
    pair: Tuple[int, int],
) -> List[Tuple[int, int, float]]:
    """k-wise の3者対応を、指定された2エージェント間の対応に変換します。"""

    pairwise_matches = []
    for d0_id, d1_id, d2_id, score in kwise_matches:
        if pair == (0, 1):
            pairwise_matches.append((d0_id, d1_id, score))
        elif pair == (0, 2):
            pairwise_matches.append((d0_id, d2_id, score))
        elif pair == (1, 2):
            pairwise_matches.append((d1_id, d2_id, score))
        else:
            raise ValueError("pair must be (0, 1), (0, 2), or (1, 2)")
    return pairwise_matches


def evaluate_kwise_as_pairwise(
    dets_i: List[Detection],
    dets_j: List[Detection],
    kwise_matches: List[Tuple[int, int, int, float]],
    pair: Tuple[int, int],
) -> Dict[str, float]:
    """k-wise matching 結果を、指定 pair の pairwise matching として評価します。"""

    pairwise_matches = kwise_to_pairwise_matches(kwise_matches, pair=pair)
    return evaluate_pairwise_matching(dets_i, dets_j, pairwise_matches)


def run_roco_all_pairs(
    detections_by_agent: Dict[int, List[Detection]],
    agent_pairs: List[Tuple[int, int]],
    params: Dict[str, float] = None,
) -> Dict[Tuple[int, int], Dict[str, object]]:
    """指定された agent pair すべてで RoCo-style matching を実行します。"""

    if params is None:
        params = ROCO_PARAMS

    results = {}
    for i, j in agent_pairs:
        dets_i = detections_by_agent[i]
        dets_j = detections_by_agent[j]
        matches = roco_pairwise_matching(dets_i, dets_j, **params)
        results[(i, j)] = {
            "matches": matches,
            "evaluation": evaluate_pairwise_matching(dets_i, dets_j, matches),
        }
    return results


def run_pairwise_rrwm_all_pairs(
    detections_by_agent: Dict[int, List[Detection]],
    agent_pairs: List[Tuple[int, int]],
    params: Dict[str, float] = None,
) -> Dict[Tuple[int, int], Dict[str, object]]:
    """指定された agent pair すべてで pairwise RRWM matching を実行します。"""

    if params is None:
        params = RRWM_PARAMS

    results = {}
    for i, j in agent_pairs:
        dets_i = detections_by_agent[i]
        dets_j = detections_by_agent[j]
        matches, x, k_mat, candidates = pairwise_rrwm_matching(dets_i, dets_j, **params)
        results[(i, j)] = {
            "matches": matches,
            "evaluation": evaluate_pairwise_matching(dets_i, dets_j, matches),
            "soft_score": x,
            "affinity_matrix": k_mat,
            "candidates": candidates,
        }
    return results


def print_matches_with_true_ids(
    title: str,
    dets_i: List[Detection],
    dets_j: List[Detection],
    matches: List[Tuple[int, int, float]],
) -> None:
    """pairwise matching 結果を true_obj_id 付きで表示します。"""

    dets_i_dict = {det.det_id: det for det in dets_i}
    dets_j_dict = {det.det_id: det for det in dets_j}

    print(f"\n=== {title} ===")
    print("matches: det_i -> det_j, score")
    for det_i_id, det_j_id, score in matches:
        obj_i = dets_i_dict[det_i_id].true_obj_id
        obj_j = dets_j_dict[det_j_id].true_obj_id
        print(
            f"  {det_i_id} -> {det_j_id}, "
            f"score={score:.5f}, "
            f"true_obj: {obj_i} - {obj_j}"
        )


def print_kwise_matches_with_true_ids(
    title: str,
    dets_0: List[Detection],
    dets_1: List[Detection],
    dets_2: List[Detection],
    matches: List[Tuple[int, int, int, float]],
) -> None:
    """k-wise matching 結果を true_obj_id 付きで表示します。"""

    dets_0_dict = {det.det_id: det for det in dets_0}
    dets_1_dict = {det.det_id: det for det in dets_1}
    dets_2_dict = {det.det_id: det for det in dets_2}

    print(f"\n=== {title} ===")
    print("matches: det0 -> det1 -> det2, score")
    for d0_id, d1_id, d2_id, score in matches:
        obj0 = dets_0_dict[d0_id].true_obj_id
        obj1 = dets_1_dict[d1_id].true_obj_id
        obj2 = dets_2_dict[d2_id].true_obj_id
        print(
            f"  {d0_id} -> {d1_id} -> {d2_id}, "
            f"score={score:.5f}, "
            f"true_obj: {obj0} - {obj1} - {obj2}"
        )


def print_method_comparison_table(
    agent_pairs: List[Tuple[int, int]],
    roco_results: Dict[Tuple[int, int], Dict[str, object]],
    rrwm_results: Dict[Tuple[int, int], Dict[str, object]],
    kwise_matches: List[Tuple[int, int, int, float]],
    detections_by_agent: Dict[int, List[Detection]],
) -> None:
    """RoCo-style、Pairwise RRWM、k-wise RRWM の性能比較表を表示します。"""

    print("\n" + "=" * 110)
    print("Matching Performance Comparison")
    print("=" * 110)
    print(
        f"{'Method':<22}"
        f"{'Pair':<12}"
        f"{'TP':<8}"
        f"{'FP':<8}"
        f"{'FN':<8}"
        f"{'Precision':<14}"
        f"{'Recall':<14}"
    )
    print("-" * 110)

    for method_name, results in [
        ("RoCo-style", roco_results),
        ("Pairwise RRWM", rrwm_results),
    ]:
        for i, j in agent_pairs:
            eval_result = results[(i, j)]["evaluation"]
            tp = eval_result["correct"]
            fp = eval_result["predicted_matches"] - tp
            fn = eval_result["gt_matches"] - tp
            print(
                f"{method_name:<22}"
                f"{f'{i}-{j}':<12}"
                f"{tp:<8}"
                f"{fp:<8}"
                f"{fn:<8}"
                f"{eval_result['precision']:<14.4f}"
                f"{eval_result['recall']:<14.4f}"
            )
        print()

    for i, j in agent_pairs:
        eval_result = evaluate_kwise_as_pairwise(
            detections_by_agent[i],
            detections_by_agent[j],
            kwise_matches,
            pair=(i, j),
        )
        tp = eval_result["correct"]
        fp = eval_result["predicted_matches"] - tp
        fn = eval_result["gt_matches"] - tp
        print(
            f"{'k-wise MGM RRWM':<22}"
            f"{f'{i}-{j}':<12}"
            f"{tp:<8}"
            f"{fp:<8}"
            f"{fn:<8}"
            f"{eval_result['precision']:<14.4f}"
            f"{eval_result['recall']:<14.4f}"
        )

    print("=" * 110)
