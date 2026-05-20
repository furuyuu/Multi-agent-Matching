import matplotlib.pyplot as plt
import numpy as np

from src.simulation import Detection, ObjectGT


def save_figure(fig, save_path=None, dpi=200):
    """save_path が指定された場合だけ、図をファイルとして保存します。"""

    if save_path is None:
        return
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")


def draw_pose(ax, x, y, theta, label=None, marker="o"):
    """位置と向きを持つ点を描画します。

    点が位置、矢印が姿勢 theta を表します。
    """

    # 位置を点として描画します。
    ax.scatter([x], [y], marker=marker, s=90)

    # theta 方向に短い矢印を描き、向きを見えるようにします。
    ax.arrow(
        x,
        y,
        2.5 * np.cos(theta),
        2.5 * np.sin(theta),
        head_width=0.8,
        length_includes_head=True,
    )

    # label が指定されている場合だけ、点の近くに文字を置きます。
    if label is not None:
        ax.text(x + 0.6, y + 0.6, label, fontsize=9)


def plot_true_scene(agents_gt, objects_gt, save_path=None):
    """真のシーンを描画します。

    表示するもの:
    - 真の物体位置・姿勢
    - 真のエージェント位置・姿勢
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # 真の物体を四角 marker で描画します。
    for obj in objects_gt:
        draw_pose(ax, obj.x, obj.y, obj.theta, label=f"O{obj.obj_id}", marker="s")

    # 真のエージェントを三角 marker で描画します。
    for ag in agents_gt:
        draw_pose(ax, ag.x, ag.y, ag.theta, label=f"A{ag.agent_id}", marker="^")

    ax.set_title("True scene: ground-truth agents and objects")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    # x, y のスケールを揃え、距離感が歪まないようにします。
    ax.axis("equal")
    ax.grid(True)
    save_figure(fig, save_path)
    plt.show()


def plot_detected_scene(agents_gt, agent_pose_est, detections_by_agent, save_path=None):
    """各エージェントの検出結果を global 座標上に描画します。

    表示するもの:
    - 真のエージェント位置
    - ノイズ付き自己位置
    - 各エージェントが検出した物体位置
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # 真のエージェント位置を小さめの三角 marker で描画します。
    for ag in agents_gt:
        ax.scatter([ag.x], [ag.y], marker="^", s=50)
        ax.text(ag.x + 0.5, ag.y + 0.5, f"A{ag.agent_id} true", fontsize=8)

    # ノイズ付き自己位置を x marker と矢印で描画します。
    for agent_id, pose in agent_pose_est.items():
        x, y, theta = pose
        draw_pose(ax, x, y, theta, label=f"A{agent_id} noisy", marker="x")

    # 各エージェントの検出点を agent_id ごとにまとめて描画します。
    for agent_id, dets in detections_by_agent.items():
        xs = [d.x_global_est for d in dets]
        ys = [d.y_global_est for d in dets]
        ax.scatter(xs, ys, marker="o", s=45, label=f"agent {agent_id} detections")

        # 各検出点に評価用ラベルを表示します。
        # 真の物体由来なら O{id}、outlier なら out と表示します。
        for d in dets:
            label = f"O{d.true_obj_id}" if d.true_obj_id >= 0 else "out"
            ax.text(d.x_global_est + 0.3, d.y_global_est + 0.3, label, fontsize=8)

    ax.set_title("Observed scene: noisy projected detections")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    # global 座標上の距離関係を見るため、x, y のスケールを揃えます。
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    save_figure(fig, save_path)
    plt.show()


def plot_pairwise_matching_result(
    dets_i: list[Detection],
    dets_j: list[Detection],
    matches: list[tuple[int, int, float]],
    agent_i: int,
    agent_j: int,
    title: str = "Pairwise matching result",
    params: dict = None,
    objects_gt: list[ObjectGT] = None,
    save_path=None,
):
    """2エージェント間のマッチング結果を可視化します。

    緑線は正しい対応、赤線は誤対応、青点線は本来対応すべき未検出対応です。
    右側には、必要に応じてハイパーパラメータを表示します。
    """

    fig, ax = plt.subplots(figsize=(12, 7))
    dets_i_dict = {det.det_id: det for det in dets_i}
    dets_j_dict = {det.det_id: det for det in dets_j}

    # 正解対応を true_obj_id から作ります。
    gt_pairs = set()
    for di in dets_i:
        if di.true_obj_id < 0:
            continue
        for dj in dets_j:
            if dj.true_obj_id < 0:
                continue
            if di.true_obj_id == dj.true_obj_id:
                gt_pairs.add((di.det_id, dj.det_id))

    predicted_pairs = {(di, dj) for di, dj, _ in matches}
    true_positive_pairs = predicted_pairs & gt_pairs
    false_positive_pairs = predicted_pairs - gt_pairs
    false_negative_pairs = gt_pairs - predicted_pairs

    # 2つのエージェントの検出点を marker を変えて描画します。
    ax.scatter(
        [d.x_global_est for d in dets_i],
        [d.y_global_est for d in dets_i],
        marker="o",
        s=80,
        label=f"Agent {agent_i}",
    )
    ax.scatter(
        [d.x_global_est for d in dets_j],
        [d.y_global_est for d in dets_j],
        marker="s",
        s=80,
        label=f"Agent {agent_j}",
    )

    # 必要に応じて真の物体位置を、検出点とは別の色・marker で重ねます。
    if objects_gt is not None:
        visible_true_obj_ids = {
            d.true_obj_id
            for d in [*dets_i, *dets_j]
            if d.true_obj_id >= 0
        }
        visible_objects_gt = [
            obj for obj in objects_gt if obj.obj_id in visible_true_obj_ids
        ]
        if visible_objects_gt:
            ax.scatter(
                [obj.x for obj in visible_objects_gt],
                [obj.y for obj in visible_objects_gt],
                marker="*",
                s=180,
                color="magenta",
                edgecolors="black",
                linewidths=0.8,
                label="True object position",
                zorder=5,
            )
            for obj in visible_objects_gt:
                ax.text(
                    obj.x + 0.3,
                    obj.y - 0.7,
                    f"GT-O{obj.obj_id}",
                    fontsize=8,
                    color="magenta",
                )

    # 各検出点に det_id と true_obj_id を表示します。
    for d in dets_i:
        obj_label = f"O{d.true_obj_id}" if d.true_obj_id >= 0 else "out"
        ax.text(
            d.x_global_est + 0.3,
            d.y_global_est + 0.3,
            f"A{agent_i}-d{d.det_id}/{obj_label}",
            fontsize=8,
        )

    for d in dets_j:
        obj_label = f"O{d.true_obj_id}" if d.true_obj_id >= 0 else "out"
        ax.text(
            d.x_global_est + 0.3,
            d.y_global_est + 0.3,
            f"A{agent_j}-d{d.det_id}/{obj_label}",
            fontsize=8,
        )

    # FN: 本来対応すべきだったが、予測されなかった対応です。
    for det_i_id, det_j_id in false_negative_pairs:
        di = dets_i_dict[det_i_id]
        dj = dets_j_dict[det_j_id]
        ax.plot(
            [di.x_global_est, dj.x_global_est],
            [di.y_global_est, dj.y_global_est],
            color="blue",
            linestyle=":",
            linewidth=2,
            alpha=0.7,
            label="FN (missed match)",
        )

    # TP: 正しく予測できた対応です。
    for det_i_id, det_j_id in true_positive_pairs:
        di = dets_i_dict[det_i_id]
        dj = dets_j_dict[det_j_id]
        ax.plot(
            [di.x_global_est, dj.x_global_est],
            [di.y_global_est, dj.y_global_est],
            color="green",
            linewidth=3,
            alpha=0.9,
            label="TP (correct match)",
        )

    # FP: 予測したが、true_obj_id が一致しなかった対応です。
    for det_i_id, det_j_id in false_positive_pairs:
        di = dets_i_dict[det_i_id]
        dj = dets_j_dict[det_j_id]
        ax.plot(
            [di.x_global_est, dj.x_global_est],
            [di.y_global_est, dj.y_global_est],
            color="red",
            linewidth=3,
            alpha=0.9,
            label="FP (wrong match)",
        )

    # 予測された対応の線の中央に score を表示します。
    for det_i_id, det_j_id, score in matches:
        di = dets_i_dict[det_i_id]
        dj = dets_j_dict[det_j_id]
        mx = (di.x_global_est + dj.x_global_est) / 2
        my = (di.y_global_est + dj.y_global_est) / 2
        ax.text(mx, my, f"{score:.2e}", fontsize=8)

    tp = len(true_positive_pairs)
    fp = len(false_positive_pairs)
    fn = len(false_negative_pairs)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    summary_text = (
        f"TP: {tp}\n"
        f"FP: {fp}\n"
        f"FN: {fn}\n"
        f"Precision: {precision:.3f}\n"
        f"Recall: {recall:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    if params is not None:
        param_text = "Hyperparameters\n" + "\n".join(
            [f"{key}: {value}" for key, value in params.items()]
        )
        ax.text(
            1.03,
            0.98,
            param_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

    # 同じ label が何度も出るため、凡例は重複を除いて表示します。
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="lower right")

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    plt.subplots_adjust(right=0.75)
    save_figure(fig, save_path)
    plt.show()
