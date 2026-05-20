from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np

from src.graph_matching import (
    ROCO_PARAMS,
    evaluate_pairwise_matching,
    roco_pairwise_matching,
)
from src.simulation import Detection, local_to_global, wrap_angle


Pose2D = Tuple[float, float, float]
PairwiseMatches = Dict[Tuple[int, int], List[Tuple[int, int, float]]]


ROCO_ITERATIVE_PARAMS = {
    "max_iter": 10,
    "damping": 0.8,
    "pose_tol": 1e-3,
    "min_matches_for_pose": 2,
}


def transform_detection_with_pose(det: Detection, agent_pose: Pose2D) -> Detection:
    """local 観測値を指定 pose で global 座標へ投影し直します。"""

    gx_est, gy_est = local_to_global(det.x_local, det.y_local, agent_pose)
    gtheta_est = wrap_angle(det.theta_local + agent_pose[2])
    return replace(
        det,
        x_global_est=gx_est,
        y_global_est=gy_est,
        theta_global_est=gtheta_est,
    )


def transform_detections_with_poses(
    detections_by_agent: Dict[int, List[Detection]],
    agent_poses: Dict[int, Pose2D],
) -> Dict[int, List[Detection]]:
    """全エージェントの detection を、現在の推定 pose で global へ投影し直します。"""

    return {
        agent_id: [
            transform_detection_with_pose(det, agent_poses[agent_id])
            for det in detections
        ]
        for agent_id, detections in detections_by_agent.items()
    }


def estimate_pose_from_local_to_global_points(
    local_points: np.ndarray,
    global_points: np.ndarray,
    current_pose: Pose2D,
    damping: float = 1.0,
    min_matches: int = 2,
) -> Tuple[Pose2D, bool]:
    """local 点群を global 点群へ合わせる 2D rigid transform を推定します。

    2点以上ある場合は Procrustes/Kabsch により回転と並進を推定します。
    1点しかない場合は現在の yaw を固定し、並進だけを合わせます。
    """

    num_points = len(local_points)
    if num_points == 0:
        return current_pose, False

    current_xy = np.array(current_pose[:2], dtype=float)
    current_theta = current_pose[2]

    if num_points < min_matches:
        if num_points != 1:
            return current_pose, False

        rotation = _rotmat(current_theta)
        estimated_xy = global_points[0] - rotation @ local_points[0]
        estimated_theta = current_theta
    else:
        src_centroid = local_points.mean(axis=0)
        dst_centroid = global_points.mean(axis=0)
        src_centered = local_points - src_centroid
        dst_centered = global_points - dst_centroid

        h_mat = src_centered.T @ dst_centered
        u_mat, _, vt_mat = np.linalg.svd(h_mat)
        rotation = vt_mat.T @ u_mat.T
        if np.linalg.det(rotation) < 0:
            vt_mat[-1, :] *= -1
            rotation = vt_mat.T @ u_mat.T

        estimated_theta = np.arctan2(rotation[1, 0], rotation[0, 0])
        estimated_xy = dst_centroid - rotation @ src_centroid

    damping = float(np.clip(damping, 0.0, 1.0))
    updated_xy = current_xy + damping * (estimated_xy - current_xy)
    updated_theta = wrap_angle(
        current_theta + damping * wrap_angle(estimated_theta - current_theta)
    )

    return (float(updated_xy[0]), float(updated_xy[1]), float(updated_theta)), True


def estimate_agent_pose_from_anchor_matches(
    anchor_detections: List[Detection],
    target_detections: List[Detection],
    matches: List[Tuple[int, int, float]],
    current_pose: Pose2D,
    damping: float = 1.0,
    min_matches: int = 2,
) -> Tuple[Pose2D, bool]:
    """anchor agent との matching から target agent の pose を推定します。"""

    anchor_by_id = {det.det_id: det for det in anchor_detections}
    target_by_id = {det.det_id: det for det in target_detections}
    local_points = []
    global_points = []

    for anchor_det_id, target_det_id, _ in matches:
        if anchor_det_id not in anchor_by_id or target_det_id not in target_by_id:
            continue
        anchor_det = anchor_by_id[anchor_det_id]
        target_det = target_by_id[target_det_id]
        local_points.append([target_det.x_local, target_det.y_local])
        global_points.append([anchor_det.x_global_est, anchor_det.y_global_est])

    return estimate_pose_from_local_to_global_points(
        np.asarray(local_points, dtype=float),
        np.asarray(global_points, dtype=float),
        current_pose=current_pose,
        damping=damping,
        min_matches=min_matches,
    )


def run_roco_matching_all_pairs(
    detections_by_agent: Dict[int, List[Detection]],
    agent_pairs: List[Tuple[int, int]],
    roco_params: Dict[str, float] = None,
) -> PairwiseMatches:
    """指定された全 agent pair で RoCo matching を実行します。"""

    if roco_params is None:
        roco_params = ROCO_PARAMS

    return {
        (i, j): roco_pairwise_matching(
            detections_by_agent[i],
            detections_by_agent[j],
            **roco_params,
        )
        for i, j in agent_pairs
    }


def estimate_object_positions_from_anchor_matches(
    detections_by_agent: Dict[int, List[Detection]],
    pairwise_matches: PairwiseMatches,
    anchor_agent_id: int = 0,
) -> Dict[str, np.ndarray]:
    """anchor detection を track とし、対応検出の平均から物体位置を推定します。"""

    tracks = {}
    anchor_detections = {
        det.det_id: det for det in detections_by_agent[anchor_agent_id]
    }

    for anchor_det_id, anchor_det in anchor_detections.items():
        points = [np.array([anchor_det.x_global_est, anchor_det.y_global_est])]

        for (i, j), matches in pairwise_matches.items():
            if i == anchor_agent_id:
                other_agent_id = j
                for det_i_id, det_j_id, _ in matches:
                    if det_i_id == anchor_det_id:
                        other_det = _get_detection_by_id(
                            detections_by_agent[other_agent_id],
                            det_j_id,
                        )
                        if other_det is not None:
                            points.append(
                                np.array(
                                    [other_det.x_global_est, other_det.y_global_est]
                                )
                            )
            elif j == anchor_agent_id:
                other_agent_id = i
                for det_i_id, det_j_id, _ in matches:
                    if det_j_id == anchor_det_id:
                        other_det = _get_detection_by_id(
                            detections_by_agent[other_agent_id],
                            det_i_id,
                        )
                        if other_det is not None:
                            points.append(
                                np.array(
                                    [other_det.x_global_est, other_det.y_global_est]
                                )
                            )

        if len(points) >= 2:
            tracks[f"A{anchor_agent_id}_d{anchor_det_id}"] = np.mean(points, axis=0)

    return tracks


def run_iterative_roco_pose_adjustment(
    detections_by_agent: Dict[int, List[Detection]],
    agent_pose_est: Dict[int, Pose2D],
    agent_pairs: List[Tuple[int, int]],
    roco_params: Dict[str, float] = None,
    anchor_agent_id: int = 0,
    max_iter: int = 10,
    damping: float = 0.8,
    pose_tol: float = 1e-3,
    min_matches_for_pose: int = 2,
) -> Dict[str, object]:
    """RoCo matching と pose adjustment を交互に実行します。

    現在は anchor_agent_id を固定し、anchor との対応から他エージェントの
    pose を更新する最小実装です。既存の matching-only 比較とは独立に使えます。
    """

    if roco_params is None:
        roco_params = ROCO_PARAMS

    current_poses = {
        agent_id: tuple(pose) for agent_id, pose in agent_pose_est.items()
    }
    history = []

    for iteration in range(max_iter):
        adjusted_detections = transform_detections_with_poses(
            detections_by_agent,
            current_poses,
        )
        pairwise_matches = run_roco_matching_all_pairs(
            adjusted_detections,
            agent_pairs,
            roco_params=roco_params,
        )

        next_poses = dict(current_poses)
        pose_deltas = {}

        for agent_id in sorted(current_poses):
            if agent_id == anchor_agent_id:
                pose_deltas[agent_id] = 0.0
                continue

            pair = (anchor_agent_id, agent_id)
            reverse_pair = (agent_id, anchor_agent_id)

            if pair in pairwise_matches:
                anchor_dets = adjusted_detections[anchor_agent_id]
                target_dets = adjusted_detections[agent_id]
                matches = pairwise_matches[pair]
            elif reverse_pair in pairwise_matches:
                anchor_dets = adjusted_detections[anchor_agent_id]
                target_dets = adjusted_detections[agent_id]
                matches = [
                    (det_j_id, det_i_id, score)
                    for det_i_id, det_j_id, score in pairwise_matches[reverse_pair]
                ]
            else:
                pose_deltas[agent_id] = 0.0
                continue

            updated_pose, did_update = estimate_agent_pose_from_anchor_matches(
                anchor_dets,
                target_dets,
                matches,
                current_pose=current_poses[agent_id],
                damping=damping,
                min_matches=min_matches_for_pose,
            )

            if did_update:
                next_poses[agent_id] = updated_pose

            pose_deltas[agent_id] = pose_update_norm(
                current_poses[agent_id],
                next_poses[agent_id],
            )

        current_poses = next_poses
        adjusted_detections = transform_detections_with_poses(
            detections_by_agent,
            current_poses,
        )
        pairwise_matches = run_roco_matching_all_pairs(
            adjusted_detections,
            agent_pairs,
            roco_params=roco_params,
        )
        pairwise_evaluations = {
            pair: evaluate_pairwise_matching(
                adjusted_detections[pair[0]],
                adjusted_detections[pair[1]],
                matches,
            )
            for pair, matches in pairwise_matches.items()
        }
        object_positions = estimate_object_positions_from_anchor_matches(
            adjusted_detections,
            pairwise_matches,
            anchor_agent_id=anchor_agent_id,
        )

        history.append(
            {
                "iteration": iteration + 1,
                "agent_poses": dict(current_poses),
                "pose_deltas": dict(pose_deltas),
                "pairwise_matches": pairwise_matches,
                "pairwise_evaluations": pairwise_evaluations,
                "object_positions": object_positions,
            }
        )

        if max(pose_deltas.values(), default=0.0) < pose_tol:
            break

    return {
        "agent_poses": current_poses,
        "detections_by_agent": transform_detections_with_poses(
            detections_by_agent,
            current_poses,
        ),
        "pairwise_matches": history[-1]["pairwise_matches"] if history else {},
        "pairwise_evaluations": history[-1]["pairwise_evaluations"] if history else {},
        "object_positions": history[-1]["object_positions"] if history else {},
        "history": history,
    }


def build_roco_iteration_dataframe(iterative_result: Dict[str, object]):
    """反復ごとの pose 変化量と matching 精度を pandas.DataFrame にします。"""

    import pandas as pd

    records = []
    for state in iterative_result["history"]:
        iteration = state["iteration"]
        max_pose_delta = max(state["pose_deltas"].values(), default=0.0)
        for pair, evaluation in state["pairwise_evaluations"].items():
            records.append(
                {
                    "iteration": iteration,
                    "pair": f"{pair[0]}-{pair[1]}",
                    "max_pose_delta": max_pose_delta,
                    "tp": evaluation["correct"],
                    "predicted_matches": evaluation["predicted_matches"],
                    "gt_matches": evaluation["gt_matches"],
                    "precision": evaluation["precision"],
                    "recall": evaluation["recall"],
                }
            )

    return pd.DataFrame.from_records(records)


def build_estimated_object_position_dataframe(iterative_result: Dict[str, object]):
    """推定された物体位置を pandas.DataFrame にします。"""

    import pandas as pd

    records = []
    for track_id, position in iterative_result["object_positions"].items():
        records.append(
            {
                "track_id": track_id,
                "x_est": float(position[0]),
                "y_est": float(position[1]),
            }
        )
    return pd.DataFrame.from_records(records)


def _rotmat(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _get_detection_by_id(detections: List[Detection], det_id: int):
    for det in detections:
        if det.det_id == det_id:
            return det
    return None


def pose_update_norm(before: Pose2D, after: Pose2D) -> float:
    """pose 更新量を、並進差 + yaw 差としてスカラー化します。"""

    before_xy = np.array(before[:2])
    after_xy = np.array(after[:2])
    translation_delta = np.linalg.norm(after_xy - before_xy)
    yaw_delta = abs(wrap_angle(after[2] - before[2]))
    return float(translation_delta + yaw_delta)
