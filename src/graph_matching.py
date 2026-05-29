from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except ModuleNotFoundError:
    _scipy_linear_sum_assignment = None

from src.simulation import Detection, wrap_angle


ROCO_PARAMS = {
    "tau2": 4.0,
    "tau1": 0.3,
    "lambda_dist": 1.0,
    "neighbor_radius": 15.0,
}

RRWM_PARAMS = {
    "candidate_radius": 4.0,
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

PARTIAL_OT_PARAMS = {
    "epsilon": 8.0,
    "epsilon_decay": 0.93,
    "beta2": 0.8,
    "outer_iter": 30,
    "inner_iter": 100,
    "tol": 1e-5,
    "match_score_threshold": 1e-4,
}


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


def points_from_detections(
    detections: List[Detection],
    coordinate: str = "global",
) -> np.ndarray:
    """Detection 群を 2D 点群に変換します。"""

    if coordinate == "global":
        return np.asarray([detection_position(det) for det in detections], dtype=float)
    if coordinate == "local":
        return np.asarray([[det.x_local, det.y_local] for det in detections], dtype=float)
    raise ValueError("coordinate must be 'global' or 'local'")


def pose_from_rotation_translation(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Tuple[float, float, float]:
    """2D 回転行列・並進ベクトルを Pose2D 相当の tuple にします。"""

    theta = np.arctan2(rotation[1, 0], rotation[0, 0])
    return (float(translation[0]), float(translation[1]), float(theta))


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """点群に 2D rigid transform を適用します。"""

    return points @ rotation.T + translation


def estimate_weighted_rigid_transform_2d(
    source_points: np.ndarray,
    target_points: np.ndarray,
    transport_plan: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """transport plan で重み付けして target -> source の剛体変換を推定します。"""

    total_mass = float(np.sum(transport_plan))
    if total_mass <= 1e-12:
        return np.eye(2), np.zeros(2), False

    # transport_plan[i, j] は「source_i と target_j を対応させる重み」です。
    # 行和・列和を各点の有効な重みとして使い、対応していない外れ値の影響を小さくします。
    row_mass = np.sum(transport_plan, axis=1)
    col_mass = np.sum(transport_plan, axis=0)
    source_centroid = (source_points.T @ row_mass) / total_mass
    target_centroid = (target_points.T @ col_mass) / total_mass

    # 論文の Recovering transformation と同じく、重み付き中心を引いたあと、
    # transport plan で重み付けした cross covariance を SVD にかけます。
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    cross_cov = target_centered.T @ transport_plan.T @ source_centered

    u_mat, _, vt_mat = np.linalg.svd(cross_cov)
    correction = np.eye(2)
    correction[-1, -1] = np.linalg.det(vt_mat.T @ u_mat.T)
    rotation = vt_mat.T @ correction @ u_mat.T

    # target を source 座標へ移す変換なので、t = source_center - R target_center です。
    translation = source_centroid - rotation @ target_centroid
    return rotation, translation, True


def partial_ot_transport_plan(
    source_points: np.ndarray,
    target_points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    epsilon: float = 1.0,
    beta2: float = 1.0,
    inner_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """range constraint 付き partial optimal transport plan を計算します。

    論文の RG 制約に合わせ、各点の marginal は [0, uniform mass]、
    total mass は [0, beta2] に収めます。
    """

    num_source = len(source_points)
    num_target = len(target_points)
    if num_source == 0 or num_target == 0:
        return np.zeros((num_source, num_target))

    beta2 = float(np.clip(beta2, 0.0, 1.0))
    epsilon = max(float(epsilon), 1e-12)

    # 現在の剛体変換で target を source 側に写し、点間の二乗距離を cost にします。
    # epsilon は entropy regularization の強さで、大きいほど遠い点にも質量が流れます。
    transformed_target = transform_points(target_points, rotation, translation)
    diff = source_points[:, None, :] - transformed_target[None, :, :]
    cost = np.sum(diff * diff, axis=2)
    kernel = np.exp(-cost / epsilon)

    # 論文の実験設定と同じく、各点を一様な probability mass とみなします。
    # source/target の marginal は [0, uniform mass] に制限するため、
    # 対応先がない点は質量 0 まで落とせます。
    source_mass = np.full(num_source, 1.0 / num_source)
    target_mass = np.full(num_target, 1.0 / num_target)

    # a, b は source/target marginal を調整する scaling 係数、
    # g は transport plan 全体の mass を調整する係数です。
    # 最終的な plan は pi = g * diag(a) * K * diag(b) になります。
    a = np.ones(num_source)
    b = np.ones(num_target)
    g = 1.0
    eps = 1e-12

    for _ in range(inner_iter):
        old_a = a.copy()
        old_b = b.copy()
        old_g = g

        # source 側の marginal が各点の持つ質量を超えないように a を更新します。
        # RG[0, 1] 制約なので、上限だけ clip し、下限は 0 を許します。
        row_proposal = g * (kernel @ b)
        a = np.minimum(row_proposal, source_mass) / np.maximum(row_proposal, eps)

        # target 側も同様に、各 target 点へ流れ込む質量を uniform mass 以下にします。
        col_proposal = g * (kernel.T @ a)
        b = np.minimum(col_proposal, target_mass) / np.maximum(col_proposal, eps)

        # total transported mass を beta2 以下に制限します。
        # beta2 を小さくすると、重なっている部分だけを選びやすくなります。
        unscaled_total = float(a @ kernel @ b)
        total_proposal = g * unscaled_total
        clipped_total = min(total_proposal, beta2)
        g = clipped_total / max(unscaled_total, eps)

        # scaling 係数の変化が小さくなったら、transport plan の内側反復を止めます。
        delta = (
            np.linalg.norm(a - old_a)
            + np.linalg.norm(b - old_b)
            + abs(g - old_g)
        )
        if delta < tol:
            break

    # 連続値の soft correspondence。値が大きいほど、その点ペアに質量が流れています。
    return g * (a[:, None] * kernel * b[None, :])


def partial_ot_rigid_registration_2d(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    epsilon: float = 8.0,
    epsilon_decay: float = 0.93,
    beta2: float = 0.8,
    outer_iter: int = 30,
    inner_iter: int = 100,
    tol: float = 1e-5,
) -> Tuple[Tuple[float, float, float], np.ndarray]:
    """partial OT と weighted Procrustes を交互に解く 2D rigid registration。"""

    # initial_pose は target local 点群を source 座標に写す初期変換として使います。
    # multi-agent pose adjustment では、target agent の現在 pose 推定が入ります。
    theta = initial_pose[2]
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    translation = np.array(initial_pose[:2], dtype=float)
    transport_plan = np.zeros((len(source_points), len(target_points)))
    current_epsilon = float(epsilon)

    for _ in range(outer_iter):
        old_rotation = rotation.copy()
        old_translation = translation.copy()

        # 1. 現在の pose で点間 cost を作り、partial OT により soft correspondence を更新します。
        transport_plan = partial_ot_transport_plan(
            source_points,
            target_points,
            rotation,
            translation,
            epsilon=current_epsilon,
            beta2=beta2,
            inner_iter=inner_iter,
        )

        # 2. 得られた soft correspondence を重みとして、target -> source の剛体変換を更新します。
        estimated_rotation, estimated_translation, did_update = estimate_weighted_rigid_transform_2d(
            source_points,
            target_points,
            transport_plan,
        )
        if not did_update:
            break

        rotation = estimated_rotation
        translation = estimated_translation

        # epsilon を少しずつ小さくして、初期は大域的、後半は局所的な対応へ寄せます。
        current_epsilon *= float(epsilon_decay)

        # pose 更新が十分小さくなったら外側反復を終了します。
        delta_translation = np.linalg.norm(translation - old_translation)
        delta_rotation = np.linalg.norm(rotation - old_rotation, ord="fro")
        if delta_translation + delta_rotation < tol:
            break

    return pose_from_rotation_translation(rotation, translation), transport_plan


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
# 3. Partial optimal transport rigid registration
# ============================================================


def discretize_transport_plan(
    transport_plan: np.ndarray,
    dets_i: List[Detection],
    dets_j: List[Detection],
    score_threshold: float = 1e-4,
) -> List[Tuple[int, int, float]]:
    """transport plan を Hungarian algorithm で1対1対応に変換します。"""

    if transport_plan.size == 0 or np.all(transport_plan <= 0):
        return []

    # transport plan は多対多の soft correspondence なので、可視化・評価用には
    # Hungarian algorithm で1対1対応へ丸めます。
    row_ind, col_ind = linear_sum_assignment(-transport_plan)
    matches = []
    for r, c in zip(row_ind, col_ind):
        score = transport_plan[r, c]
        if score > score_threshold:
            matches.append((dets_i[r].det_id, dets_j[c].det_id, float(score)))
    return matches


def partial_ot_pairwise_matching(
    dets_i: List[Detection],
    dets_j: List[Detection],
    epsilon: float = 8.0,
    epsilon_decay: float = 0.93,
    beta2: float = 0.8,
    outer_iter: int = 30,
    inner_iter: int = 100,
    tol: float = 1e-5,
    match_score_threshold: float = 1e-4,
) -> Tuple[List[Tuple[int, int, float]], Tuple[float, float, float], np.ndarray]:
    """global 推定点群同士を partial OT registration し、対応を返します。"""

    # matching-only の比較では、既に noisy pose で global に投影された点同士を登録します。
    source_points = points_from_detections(dets_i, coordinate="global")
    target_points = points_from_detections(dets_j, coordinate="global")
    pose, transport_plan = partial_ot_rigid_registration_2d(
        source_points,
        target_points,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        beta2=beta2,
        outer_iter=outer_iter,
        inner_iter=inner_iter,
        tol=tol,
    )
    matches = discretize_transport_plan(
        transport_plan,
        dets_i,
        dets_j,
        score_threshold=match_score_threshold,
    )
    return matches, pose, transport_plan


def partial_ot_anchor_pose_registration(
    anchor_detections: List[Detection],
    target_detections: List[Detection],
    current_pose: Tuple[float, float, float],
    epsilon: float = 8.0,
    epsilon_decay: float = 0.93,
    beta2: float = 0.8,
    outer_iter: int = 30,
    inner_iter: int = 100,
    tol: float = 1e-5,
    match_score_threshold: float = 1e-4,
) -> Tuple[Tuple[float, float, float], List[Tuple[int, int, float]], np.ndarray]:
    """anchor global 点群へ target local 点群を partial OT で登録します。"""

    # pose adjustment では、anchor は global 点、target は local 観測点として扱います。
    # 登録で推定される target -> anchor の剛体変換が、そのまま target agent pose になります。
    source_points = points_from_detections(anchor_detections, coordinate="global")
    target_points = points_from_detections(target_detections, coordinate="local")
    estimated_pose, transport_plan = partial_ot_rigid_registration_2d(
        source_points,
        target_points,
        initial_pose=current_pose,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        beta2=beta2,
        outer_iter=outer_iter,
        inner_iter=inner_iter,
        tol=tol,
    )
    matches = discretize_transport_plan(
        transport_plan,
        anchor_detections,
        target_detections,
        score_threshold=match_score_threshold,
    )
    return estimated_pose, matches, transport_plan


# ============================================================
# 4. 3-wise matching with RRWM
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


def kwise_to_global_pairwise_matches(
    kwise_matches: List[Tuple[int, int, int, float]],
    triple: Tuple[int, int, int],
    pair: Tuple[int, int],
) -> List[Tuple[int, int, float]]:
    """global agent id の pair を、triple 内の3-wise結果から取り出します。"""

    if pair[0] not in triple or pair[1] not in triple:
        raise ValueError("pair agents must be included in triple")

    local_pair = (triple.index(pair[0]), triple.index(pair[1]))
    return kwise_to_pairwise_matches(kwise_matches, pair=local_pair)


def consistent_pairwise_matches_from_kwise_triples(
    detections_by_agent: Dict[int, List[Detection]],
    kwise_matches_by_triple: Dict[Tuple[int, int, int], List[Tuple[int, int, int, float]]],
    agent_ids: Tuple[int, ...],
    reference_agent_id: int = 0,
    score_threshold: float = 1e-4,
) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, int, float]]], Dict[int, List[Tuple[int, int, float]]]]:
    """3-wise 結果群を reference agent 経由の cycle-consistent pairwise 対応に統合します。

    Zhu et al. の k-wise MGM は、各 k-wise indicator tensor を reference graph r
    への pairwise indicator matrix X[ir] に分解し、最後に W = U^T U で全体の
    pairwise 対応を構成します。この関数はその考え方を、既存の3-wise RRWM結果に
    対する one-shot な投票集約として実装します。
    """

    if reference_agent_id not in agent_ids:
        raise ValueError("reference_agent_id must be included in agent_ids")

    ref_detections = detections_by_agent[reference_agent_id]
    ref_id_to_index = {det.det_id: idx for idx, det in enumerate(ref_detections)}
    votes = {
        agent_id: np.zeros((len(ref_detections), len(detections_by_agent[agent_id])))
        for agent_id in agent_ids
        if agent_id != reference_agent_id
    }

    for triple, kwise_matches in kwise_matches_by_triple.items():
        if reference_agent_id not in triple:
            continue

        for match in kwise_matches:
            det_ids_by_agent = {
                triple[0]: match[0],
                triple[1]: match[1],
                triple[2]: match[2],
            }
            score = float(match[3])
            ref_det_id = det_ids_by_agent[reference_agent_id]
            if ref_det_id not in ref_id_to_index:
                continue

            ref_idx = ref_id_to_index[ref_det_id]
            for agent_id in triple:
                if agent_id == reference_agent_id:
                    continue

                target_det_id = det_ids_by_agent[agent_id]
                target_id_to_index = {
                    det.det_id: idx for idx, det in enumerate(detections_by_agent[agent_id])
                }
                target_idx = target_id_to_index.get(target_det_id)
                if target_idx is not None:
                    votes[agent_id][ref_idx, target_idx] += score

    reference_matches = {}
    for agent_id, score_matrix in votes.items():
        if score_matrix.size == 0 or np.all(score_matrix <= 0):
            reference_matches[agent_id] = []
            continue

        row_ind, col_ind = linear_sum_assignment(-score_matrix)
        matches = []
        for ref_idx, target_idx in zip(row_ind, col_ind):
            score = score_matrix[ref_idx, target_idx]
            if score <= score_threshold:
                continue

            matches.append(
                (
                    ref_detections[ref_idx].det_id,
                    detections_by_agent[agent_id][target_idx].det_id,
                    float(score),
                )
            )
        reference_matches[agent_id] = matches

    pairwise_matches = {}
    for i, j in combinations(agent_ids, 2):
        if i == reference_agent_id:
            pairwise_matches[(i, j)] = list(reference_matches.get(j, []))
            continue
        if j == reference_agent_id:
            pairwise_matches[(i, j)] = [
                (target_id, ref_id, score)
                for ref_id, target_id, score in reference_matches.get(i, [])
            ]
            continue

        i_by_ref = {
            ref_id: (target_id, score)
            for ref_id, target_id, score in reference_matches.get(i, [])
        }
        j_by_ref = {
            ref_id: (target_id, score)
            for ref_id, target_id, score in reference_matches.get(j, [])
        }
        composed = []
        for ref_id in sorted(set(i_by_ref) & set(j_by_ref)):
            i_det_id, i_score = i_by_ref[ref_id]
            j_det_id, j_score = j_by_ref[ref_id]
            composed.append((i_det_id, j_det_id, float(min(i_score, j_score))))
        pairwise_matches[(i, j)] = composed

    return pairwise_matches, reference_matches


def run_consistent_kwise_rrwm_matching(
    detections_by_agent: Dict[int, List[Detection]],
    agent_ids: Tuple[int, ...],
    kwise_params: Dict[str, float] = None,
    reference_agent_id: int = 0,
    score_threshold: float = 1e-4,
) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, int, float]]], Dict[str, object]]:
    """全3-agent triple の3-wise RRWM結果を reference 経由で一貫化します。"""

    if kwise_params is None:
        kwise_params = KWISE_PARAMS

    triples = list(combinations(agent_ids, 3))
    kwise_results_by_triple = {}
    kwise_matches_by_triple = {}

    for triple in triples:
        matches, soft_score, affinity_matrix, candidates = kwise_rrwm_matching_3agents(
            detections_by_agent[triple[0]],
            detections_by_agent[triple[1]],
            detections_by_agent[triple[2]],
            **kwise_params,
        )
        kwise_results_by_triple[triple] = {
            "matches": matches,
            "evaluation": evaluate_kwise_matching_3agents(
                detections_by_agent[triple[0]],
                detections_by_agent[triple[1]],
                detections_by_agent[triple[2]],
                matches,
            ),
            "soft_score": soft_score,
            "affinity_matrix": affinity_matrix,
            "candidates": candidates,
        }
        kwise_matches_by_triple[triple] = matches

    pairwise_matches, reference_matches = consistent_pairwise_matches_from_kwise_triples(
        detections_by_agent,
        kwise_matches_by_triple,
        agent_ids=agent_ids,
        reference_agent_id=reference_agent_id,
        score_threshold=score_threshold,
    )

    return pairwise_matches, {
        "kwise_results_by_triple": kwise_results_by_triple,
        "reference_matches": reference_matches,
        "reference_agent_id": reference_agent_id,
    }


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

    comparison_df = build_method_comparison_dataframe(
        agent_pairs=agent_pairs,
        roco_results=roco_results,
        rrwm_results=rrwm_results,
        kwise_matches=kwise_matches,
        detections_by_agent=detections_by_agent,
    )

    print("\n" + "=" * 110)
    print("Matching Performance Comparison")
    print("=" * 110)
    print(comparison_df.to_string(index=False))
    print("=" * 110)


def build_method_comparison_records(
    agent_pairs: List[Tuple[int, int]],
    roco_results: Dict[Tuple[int, int], Dict[str, object]],
    rrwm_results: Dict[Tuple[int, int], Dict[str, object]],
    kwise_matches: List[Tuple[int, int, int, float]],
    detections_by_agent: Dict[int, List[Detection]],
) -> List[Dict[str, object]]:
    """3手法の比較結果を、表にしやすい record list として作ります。"""

    records = []

    for method_name, results in [
        ("RoCo-style", roco_results),
        ("Pairwise RRWM", rrwm_results),
    ]:
        for i, j in agent_pairs:
            eval_result = results[(i, j)]["evaluation"]
            tp = int(eval_result["correct"])
            predicted = int(eval_result["predicted_matches"])
            gt = int(eval_result["gt_matches"])
            records.append(
                {
                    "method": method_name,
                    "pair": f"{i}-{j}",
                    "tp": tp,
                    "fp": predicted - tp,
                    "fn": gt - tp,
                    "gt_matches": gt,
                    "predicted_matches": predicted,
                    "precision": float(eval_result["precision"]),
                    "recall": float(eval_result["recall"]),
                }
            )

    for i, j in agent_pairs:
        eval_result = evaluate_kwise_as_pairwise(
            detections_by_agent[i],
            detections_by_agent[j],
            kwise_matches,
            pair=(i, j),
        )
        tp = int(eval_result["correct"])
        predicted = int(eval_result["predicted_matches"])
        gt = int(eval_result["gt_matches"])
        records.append(
            {
                "method": "k-wise MGM RRWM",
                "pair": f"{i}-{j}",
                "tp": tp,
                "fp": predicted - tp,
                "fn": gt - tp,
                "gt_matches": gt,
                "predicted_matches": predicted,
                "precision": float(eval_result["precision"]),
                "recall": float(eval_result["recall"]),
            }
        )

    return records


def build_method_comparison_dataframe(
    agent_pairs: List[Tuple[int, int]],
    roco_results: Dict[Tuple[int, int], Dict[str, object]],
    rrwm_results: Dict[Tuple[int, int], Dict[str, object]],
    kwise_matches: List[Tuple[int, int, int, float]],
    detections_by_agent: Dict[int, List[Detection]],
    sort_by_pair: bool = True,
):
    """3手法の比較結果を pandas.DataFrame として返します。

    sort_by_pair=True の場合は、各ペアごとに3手法を並べます。
    """

    import pandas as pd

    records = build_method_comparison_records(
        agent_pairs=agent_pairs,
        roco_results=roco_results,
        rrwm_results=rrwm_results,
        kwise_matches=kwise_matches,
        detections_by_agent=detections_by_agent,
    )
    columns = [
        "method",
        "pair",
        "tp",
        "fp",
        "fn",
        "gt_matches",
        "predicted_matches",
        "precision",
        "recall",
    ]
    df = pd.DataFrame.from_records(records, columns=columns)

    if sort_by_pair and not df.empty:
        pair_order = {f"{i}-{j}": order for order, (i, j) in enumerate(agent_pairs)}
        method_order = {
            "RoCo-style": 0,
            "Pairwise RRWM": 1,
            "k-wise MGM RRWM": 2,
        }
        df["_pair_order"] = df["pair"].map(pair_order)
        df["_method_order"] = df["method"].map(method_order)
        df = (
            df.sort_values(["_pair_order", "_method_order"])
            .drop(columns=["_pair_order", "_method_order"])
            .reset_index(drop=True)
        )

    return df
