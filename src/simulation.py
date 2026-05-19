from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class AgentGT:
    """エージェントの真値を表すクラス。

    x, y, theta はすべて global 座標系での真の位置・姿勢です。
    """

    agent_id: int
    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class ObjectGT:
    """物体の真値を表すクラス。

    x, y, theta は global 座標系での真の位置・姿勢です。
    length, width は物体サイズを表します。
    """

    obj_id: int
    x: float
    y: float
    theta: float
    length: float
    width: float


@dataclass(frozen=True)
class Detection:
    """各エージェントが検出した物体を表すクラス。

    true_obj_id が -1 の場合は、実在しない物体を検出した outlier です。
    local 系の値はエージェントから見た観測値、global_est はノイズ付き
    自己位置を使って global 座標へ変換した推定値です。
    """

    det_id: int
    agent_id: int
    true_obj_id: int
    x_local: float
    y_local: float
    theta_local: float
    x_global_est: float
    y_global_est: float
    theta_global_est: float


def wrap_angle(a: float) -> float:
    """角度を [-pi, pi] の範囲に正規化します。"""

    return (a + np.pi) % (2 * np.pi) - np.pi


def rotmat(theta: float) -> np.ndarray:
    """2次元回転行列 R(theta) を返します。"""

    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


def global_to_local(px: float, py: float, agent: AgentGT) -> Tuple[float, float]:
    """global 座標の点を、エージェント基準の local 座標へ変換します。"""

    # エージェント位置を原点とした相対ベクトルを作ります。
    p = np.array([px - agent.x, py - agent.y])
    # エージェントの向きを打ち消すため、-theta だけ回転します。
    return tuple(rotmat(-agent.theta) @ p)


def local_to_global(
    lx: float,
    ly: float,
    agent_pose: Tuple[float, float, float],
) -> Tuple[float, float]:
    """local 座標の点を、推定自己位置を使って global 座標へ変換します。"""

    ax, ay, atheta = agent_pose
    # local 点を推定姿勢だけ回転し、推定位置を足します。
    p = np.array([ax, ay]) + rotmat(atheta) @ np.array([lx, ly])
    return tuple(p)


def generate_true_scene(
    num_agents: int = 3,
    num_objects: int = 12,
    area_size: Tuple[float, float] = (60.0, 40.0),
    seed: int = 7,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[AgentGT], List[ObjectGT]]:
    """真のエージェント配置と真の物体配置を生成します。"""

    # rng を渡すと、呼び出し元で乱数状態を共有できます。
    # rng がない場合は seed から新しい乱数生成器を作ります。
    if rng is None:
        rng = np.random.default_rng(seed)

    # 3エージェントは手動配置します。
    # theta はエージェントの向きです。
    agents = [
        AgentGT(0, -18.0, -8.0, np.deg2rad(15)),
        AgentGT(1, 18.0, -6.0, np.deg2rad(165)),
        AgentGT(2, 0.0, 16.0, np.deg2rad(-90)),
    ]

    if num_agents > len(agents):
        raise ValueError(f"num_agents must be <= {len(agents)}")

    width, height = area_size
    objects = []
    for obj_id in range(num_objects):
        # 物体は指定領域の内側にランダム配置します。
        # サイズは車両のような長方形物体を想定しています。
        objects.append(
            ObjectGT(
                obj_id=obj_id,
                x=rng.uniform(-width / 2 + 5, width / 2 - 5),
                y=rng.uniform(-height / 2 + 5, height / 2 - 5),
                theta=rng.uniform(-np.pi, np.pi),
                length=rng.uniform(3.5, 5.0),
                width=rng.uniform(1.6, 2.2),
            )
        )

    return agents[:num_agents], objects


def generate_detections(
    agents_gt: List[AgentGT],
    objects_gt: List[ObjectGT],
    sensing_range: float = 32.0,
    fov_deg: float = 150.0,
    detection_prob: float = 0.85,
    object_pos_noise_std: float = 0.7,
    object_yaw_noise_std_deg: float = 4.0,
    agent_pos_noise_std: float = 1.0,
    agent_yaw_noise_std_deg: float = 3.0,
    num_outliers_per_agent: int = 2,
    seed: int = 7,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, List[Detection]]]:
    """各エージェントのノイズ付き検出結果を生成します。

    処理の流れ:
    1. 各エージェントにノイズ付き自己位置を与える
    2. 真の物体をエージェント local 座標へ変換する
    3. センシング範囲・視野角内の物体だけを検出候補にする
    4. 検出確率に従って一部の物体を検出する
    5. local 観測値にノイズを加える
    6. ノイズ付き自己位置を使って global 座標へ投影する
    7. outlier、つまり誤検出を追加する
    """

    # generate_true_scene と同じ rng を渡すと、以前の notebook のように
    # シーン生成後の乱数状態から検出生成を続けられます。
    if rng is None:
        rng = np.random.default_rng(seed)
    # 視野角を degree から radian に変換します。
    fov = np.deg2rad(fov_deg)

    # agent_pose_est: agent_id -> ノイズ付き自己位置 (x, y, theta)
    agent_pose_est = {}
    # detections_by_agent: agent_id -> そのエージェントが得た Detection のリスト
    detections_by_agent = {}

    for ag in agents_gt:
        # 真の自己位置にノイズを足し、エージェントが持つ推定自己位置を作ります。
        ax_noisy = ag.x + rng.normal(0, agent_pos_noise_std)
        ay_noisy = ag.y + rng.normal(0, agent_pos_noise_std)
        atheta_noisy = wrap_angle(
            ag.theta + rng.normal(0, np.deg2rad(agent_yaw_noise_std_deg))
        )
        agent_pose_est[ag.agent_id] = (ax_noisy, ay_noisy, atheta_noisy)

        dets = []
        det_id = 0

        for obj in objects_gt:
            # 真の物体位置を、現在のエージェントから見た local 座標へ変換します。
            lx, ly = global_to_local(obj.x, obj.y, ag)

            # 距離と方位角から、センサ範囲・視野角内かどうかを判定します。
            dist = np.hypot(lx, ly)
            bearing = np.arctan2(ly, lx)
            visible = dist <= sensing_range and abs(bearing) <= fov / 2

            # 見えていない物体、または検出漏れした物体は Detection にしません。
            if not visible or rng.random() >= detection_prob:
                continue

            # 物体検出器の中心位置推定誤差を local 座標上で加えます。
            lx_n = lx + rng.normal(0, object_pos_noise_std)
            ly_n = ly + rng.normal(0, object_pos_noise_std)

            # local 座標系で見た物体向きにも検出ノイズを加えます。
            ltheta = wrap_angle(obj.theta - ag.theta)
            ltheta_n = wrap_angle(
                ltheta + rng.normal(0, np.deg2rad(object_yaw_noise_std_deg))
            )

            # 重要: 真の自己位置ではなく、ノイズ付き自己位置で global に戻します。
            # そのため global_est には物体検出ノイズと自己位置推定ノイズの両方が含まれます。
            gx_est, gy_est = local_to_global(
                lx_n,
                ly_n,
                agent_pose_est[ag.agent_id],
            )
            gtheta_est = wrap_angle(ltheta_n + agent_pose_est[ag.agent_id][2])

            dets.append(
                Detection(
                    det_id=det_id,
                    agent_id=ag.agent_id,
                    true_obj_id=obj.obj_id,
                    x_local=lx_n,
                    y_local=ly_n,
                    theta_local=ltheta_n,
                    x_global_est=gx_est, # ノイズを含む
                    y_global_est=gy_est, # ノイズを含む
                    theta_global_est=gtheta_est, # ノイズを含む
                )
            )
            det_id += 1

        for _ in range(num_outliers_per_agent):
            # 実在しない物体を、local 座標上のランダムな位置に生成します。
            lx_o = rng.uniform(5.0, sensing_range)
            ly_o = rng.uniform(-sensing_range / 2, sensing_range / 2)
            theta_o = rng.uniform(-np.pi, np.pi)

            # outlier も通常の検出と同じく、ノイズ付き自己位置で global に投影します。
            gx_est, gy_est = local_to_global(
                lx_o,
                ly_o,
                agent_pose_est[ag.agent_id],
            )
            gtheta_est = wrap_angle(theta_o + agent_pose_est[ag.agent_id][2])

            dets.append(
                Detection(
                    det_id=det_id,
                    agent_id=ag.agent_id,
                    # -1 は outlier を表す評価用ラベルです。
                    true_obj_id=-1,
                    x_local=lx_o,
                    y_local=ly_o,
                    theta_local=theta_o,
                    x_global_est=gx_est,
                    y_global_est=gy_est,
                    theta_global_est=gtheta_est,
                )
            )
            det_id += 1

        detections_by_agent[ag.agent_id] = dets

    return agent_pose_est, detections_by_agent


def count_detection_labels(
    detections_by_agent: Dict[int, List[Detection]],
) -> Dict[int, Tuple[int, int, int]]:
    """各エージェントの検出数、inlier 数、outlier 数を数えます。"""

    counts = {}
    for agent_id, dets in detections_by_agent.items():
        inliers = sum(d.true_obj_id >= 0 for d in dets)
        outliers = sum(d.true_obj_id < 0 for d in dets)
        counts[agent_id] = (len(dets), inliers, outliers)
    return counts
