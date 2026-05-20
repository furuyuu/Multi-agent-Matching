from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.graph_matching import (
    ROCO_PARAMS,
    evaluate_pairwise_matching,
    roco_pairwise_matching,
)
from src.simulation import AgentGT, Detection, ObjectGT, local_to_global, wrap_angle


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

    object_poses, _ = estimate_object_poses_from_anchor_matches(
        detections_by_agent,
        pairwise_matches,
        anchor_agent_id=anchor_agent_id,
    )
    return {
        track_id: np.array([pose[0], pose[1]], dtype=float)
        for track_id, pose in object_poses.items()
    }


def estimate_object_poses_from_anchor_matches(
    detections_by_agent: Dict[int, List[Detection]],
    pairwise_matches: PairwiseMatches,
    anchor_agent_id: int = 0,
) -> Tuple[Dict[str, Pose2D], Dict[str, int]]:
    """anchor detection を track とし、対応検出の平均から物体 pose を推定します。"""

    tracks = {}
    track_true_object_ids = {}
    anchor_detections = {
        det.det_id: det for det in detections_by_agent[anchor_agent_id]
    }

    for anchor_det_id, anchor_det in anchor_detections.items():
        points = [np.array([anchor_det.x_global_est, anchor_det.y_global_est])]
        angles = [anchor_det.theta_global_est]

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
                            angles.append(other_det.theta_global_est)
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
                            angles.append(other_det.theta_global_est)

        if len(points) >= 2:
            xy_mean = np.mean(points, axis=0)
            theta_mean = _circular_mean(angles)
            track_id = f"A{anchor_agent_id}_d{anchor_det_id}"
            tracks[track_id] = (
                float(xy_mean[0]),
                float(xy_mean[1]),
                float(theta_mean),
            )
            track_true_object_ids[track_id] = anchor_det.true_obj_id

    return tracks, track_true_object_ids


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
        object_poses, track_true_object_ids = estimate_object_poses_from_anchor_matches(
            adjusted_detections,
            pairwise_matches,
            anchor_agent_id=anchor_agent_id,
        )
        object_positions = {
            track_id: np.array([pose[0], pose[1]], dtype=float)
            for track_id, pose in object_poses.items()
        }

        history.append(
            {
                "iteration": iteration + 1,
                "agent_poses": dict(current_poses),
                "pose_deltas": dict(pose_deltas),
                "pairwise_matches": pairwise_matches,
                "pairwise_evaluations": pairwise_evaluations,
                "object_poses": object_poses,
                "object_positions": object_positions,
                "track_true_object_ids": track_true_object_ids,
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
        "object_poses": history[-1]["object_poses"] if history else {},
        "object_positions": history[-1]["object_positions"] if history else {},
        "track_true_object_ids": history[-1]["track_true_object_ids"] if history else {},
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


def evaluate_iterative_roco_pose_errors(
    iterative_result: Dict[str, object],
    agents_gt: List[AgentGT],
    objects_gt: List[ObjectGT],
) -> Dict[str, object]:
    """最終的な agent/object pose 推定を真値と比較します。

    object は anchor detection 由来の track ごとに評価します。outlier track や
    真値が見つからない track は object_errors から除外されます。
    """

    agent_errors = evaluate_agent_pose_errors(
        iterative_result.get("agent_poses", {}),
        agents_gt,
    )
    object_errors = evaluate_object_pose_errors(
        iterative_result.get("object_poses", {}),
        iterative_result.get("track_true_object_ids", {}),
        objects_gt,
    )

    return {
        "summary": {
            "overall": _summarize_pose_errors(
                list(agent_errors.values()) + list(object_errors.values())
            ),
            "agents": _summarize_pose_errors(agent_errors.values()),
            "objects": _summarize_pose_errors(object_errors.values()),
        },
        "agent_errors": agent_errors,
        "object_errors": object_errors,
    }


def evaluate_iterative_roco_pose_error_history(
    iterative_result: Dict[str, object],
    agents_gt: List[AgentGT],
    objects_gt: List[ObjectGT],
) -> List[Dict[str, object]]:
    """各 iteration の agent/object pose 推定を真値と比較します。"""

    results = []
    for state in iterative_result.get("history", []):
        state_result = evaluate_iterative_roco_pose_errors(
            state,
            agents_gt,
            objects_gt,
        )
        state_result["iteration"] = state["iteration"]
        state_result["max_pose_delta"] = max(
            state.get("pose_deltas", {}).values(),
            default=0.0,
        )
        results.append(state_result)
    return results


def evaluate_agent_pose_errors(
    agent_poses: Dict[int, Pose2D],
    agents_gt: List[AgentGT],
) -> Dict[int, Dict[str, float]]:
    """agent pose 推定を真値と比較します。"""

    gt_by_id = {agent.agent_id: agent for agent in agents_gt}
    errors = {}
    for agent_id, estimated_pose in agent_poses.items():
        gt = gt_by_id.get(agent_id)
        if gt is None:
            continue
        errors[agent_id] = {
            "agent_id": agent_id,
            **_pose_error_record(
                estimated_pose,
                (gt.x, gt.y, gt.theta),
            ),
        }
    return errors


def evaluate_object_pose_errors(
    object_poses: Dict[str, Pose2D],
    track_true_object_ids: Dict[str, int],
    objects_gt: List[ObjectGT],
) -> Dict[str, Dict[str, float]]:
    """object pose 推定を、track に紐づく true_obj_id の真値と比較します。"""

    gt_by_id = {obj.obj_id: obj for obj in objects_gt}
    errors = {}
    for track_id, estimated_pose in object_poses.items():
        true_obj_id = track_true_object_ids.get(track_id, -1)
        gt = gt_by_id.get(true_obj_id)
        if gt is None:
            continue
        errors[track_id] = {
            "track_id": track_id,
            "true_obj_id": true_obj_id,
            **_pose_error_record(
                estimated_pose,
                (gt.x, gt.y, gt.theta),
            ),
        }
    return errors


def build_pose_error_summary_dataframe(pose_error_result: Dict[str, object]):
    """overall / agents / objects の pose 誤差 summary を pandas.DataFrame にします。"""

    import pandas as pd

    records = []
    for target, metrics in pose_error_result["summary"].items():
        records.append({"target": target, **metrics})
    return pd.DataFrame.from_records(records)


def build_pose_error_history_dataframe(pose_error_history: List[Dict[str, object]]):
    """iteration ごとの overall / agents / objects pose 誤差を pandas.DataFrame にします。"""

    import pandas as pd

    records = []
    for state_result in pose_error_history:
        iteration = state_result["iteration"]
        max_pose_delta = state_result["max_pose_delta"]
        for target, metrics in state_result["summary"].items():
            records.append(
                {
                    "iteration": iteration,
                    "target": target,
                    "max_pose_delta": max_pose_delta,
                    **metrics,
                }
            )
    return pd.DataFrame.from_records(records)


def build_agent_pose_error_dataframe(pose_error_result: Dict[str, object]):
    """agent ごとの pose 誤差を pandas.DataFrame にします。"""

    import pandas as pd

    return pd.DataFrame.from_records(list(pose_error_result["agent_errors"].values()))


def build_object_pose_error_dataframe(pose_error_result: Dict[str, object]):
    """object track ごとの pose 誤差を pandas.DataFrame にします。"""

    import pandas as pd

    return pd.DataFrame.from_records(list(pose_error_result["object_errors"].values()))


def _rotmat(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _get_detection_by_id(detections: List[Detection], det_id: int):
    for det in detections:
        if det.det_id == det_id:
            return det
    return None


def _circular_mean(angles: List[float]) -> float:
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def _pose_error_record(
    estimated_pose: Pose2D,
    true_pose: Pose2D,
) -> Dict[str, float]:
    estimated_xy = np.array(estimated_pose[:2], dtype=float)
    true_xy = np.array(true_pose[:2], dtype=float)
    position_error = float(np.linalg.norm(estimated_xy - true_xy))
    yaw_error = float(abs(wrap_angle(estimated_pose[2] - true_pose[2])))

    return {
        "x_est": float(estimated_pose[0]),
        "y_est": float(estimated_pose[1]),
        "theta_est": float(estimated_pose[2]),
        "x_true": float(true_pose[0]),
        "y_true": float(true_pose[1]),
        "theta_true": float(true_pose[2]),
        "position_error": position_error,
        "yaw_error": yaw_error,
        "yaw_error_deg": float(np.rad2deg(yaw_error)),
        "pose_error": float(position_error + yaw_error),
    }


def _summarize_pose_errors(error_records) -> Dict[str, Optional[float]]:
    records = list(error_records)
    if not records:
        return {
            "count": 0,
            "mean_position_error": np.nan,
            "rmse_position_error": np.nan,
            "mean_yaw_error": np.nan,
            "mean_yaw_error_deg": np.nan,
            "mean_pose_error": np.nan,
        }

    position_errors = np.array(
        [record["position_error"] for record in records],
        dtype=float,
    )
    yaw_errors = np.array([record["yaw_error"] for record in records], dtype=float)
    pose_errors = np.array([record["pose_error"] for record in records], dtype=float)

    return {
        "count": len(records),
        "mean_position_error": float(np.mean(position_errors)),
        "rmse_position_error": float(np.sqrt(np.mean(position_errors**2))),
        "mean_yaw_error": float(np.mean(yaw_errors)),
        "mean_yaw_error_deg": float(np.rad2deg(np.mean(yaw_errors))),
        "mean_pose_error": float(np.mean(pose_errors)),
    }


def pose_update_norm(before: Pose2D, after: Pose2D) -> float:
    """pose 更新量を、並進差 + yaw 差としてスカラー化します。"""

    before_xy = np.array(before[:2])
    after_xy = np.array(after[:2])
    translation_delta = np.linalg.norm(after_xy - before_xy)
    yaw_delta = abs(wrap_angle(after[2] - before[2]))
    return float(translation_delta + yaw_delta)
